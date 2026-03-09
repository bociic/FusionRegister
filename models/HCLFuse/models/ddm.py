import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models.HCLFuse.utils as utils
from models.HCLFuse.models.unet import DiffusionUNet
from models.HCLFuse.models.decom import Encoder
from models.HCLFuse.datasets.common import YCbCr2RGB

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def optimal_transport_alignment(z_ir, z_vis, epsilon=0.2, max_iter=30):
    B, C, H, W = z_ir.shape
    z_ir_flat = z_ir.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
    z_vis_flat = z_vis.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

    cost_matrix = torch.cdist(z_ir_flat, z_vis_flat, p=2) ** 2  
    cost_matrix = torch.clamp(cost_matrix, min=1e-6, max=1e6)  

    transport_plan = torch.exp(-cost_matrix / epsilon)  
    transport_plan /= transport_plan.sum(dim=-1, keepdim=True)

    for _ in range(max_iter):
        transport_plan /= torch.clamp(transport_plan.sum(dim=-2, keepdim=True), min=1e-6)
        transport_plan /= torch.clamp(transport_plan.sum(dim=-1, keepdim=True), min=1e-6)
    x_ir_aligned = torch.bmm(transport_plan, z_ir_flat).view(B, C, H, W)  
    z_ir_aligned = 0.5 * x_ir_aligned + 0.5 * z_ir  

    return z_ir_aligned

class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DiffusionUNet(config)
        self.encoder = Encoder(config)
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]
        self.info_gradient_estimator = nn.Sequential(
                    nn.Conv2d(2*config.model.in_channels, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, config.model.in_channels, 3, padding=1)
                )
    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def physics_guided_sample(self, z_cond, b, x_ir, x_vis):
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = z_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        heat_diffusion_coef = 0.05
        structure_preservation_weight = 0.15
        physics_decay_factor = 5.0
        
        if x_ir.shape[2:] != z_cond.shape[2:]:
            x_ir = F.interpolate(x_ir, size=z_cond.shape[2:], mode='bilinear', align_corners=False)
        if x_vis.shape[2:] != z_cond.shape[2:]:
            x_vis = F.interpolate(x_vis, size=z_cond.shape[2:], mode='bilinear', align_corners=False)

        ir_grad = self.compute_gradient(x_ir)
        vis_grad = self.compute_gradient(x_vis)
        max_grad = torch.max(ir_grad, vis_grad)

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = self.Unet(torch.cat([z_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            normalized_t = torch.tensor(i / self.config.diffusion.num_diffusion_timesteps, device=x.device)
            physics_weight = torch.exp(-physics_decay_factor * normalized_t)
            physics_weight = physics_weight.view(1, 1, 1, 1)

            x0_heat = self.apply_heat_equation(x0_t, alpha=heat_diffusion_coef * physics_weight)
            
            x0_structure = self.apply_structure_preservation(
                x0_heat, 
                max_grad, 
                weight=structure_preservation_weight * physics_weight
            )
            
            x0_physics = self.apply_physical_consistency(
                x0_structure, 
                x_ir, 
                x_vis, 
                weight=0.1 * physics_weight
            )
            
            et_corrected = (xt - x0_physics * at.sqrt()) / (1 - at).sqrt()
            c1 = 0.0
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_physics + c2 * et_corrected
            xs.append(xt_next.to(x.device))
        
        return xs[-1]

    def apply_heat_equation(self, x, alpha=0.1):
        padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        laplacian = (
            padded[:, :, :-2, 1:-1] + 
            padded[:, :, 2:, 1:-1] + 
            padded[:, :, 1:-1, :-2] + 
            padded[:, :, 1:-1, 2:] - 
            4 * x
        )
        return x + alpha * laplacian

    def compute_gradient(self, x):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(pad, sobel_x, groups=x.shape[1])
        grad_y = F.conv2d(pad, sobel_y, groups=x.shape[1])
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return grad_magnitude

    def apply_structure_preservation(self, x, grad_reference, weight=0.2):
        if grad_reference.shape[2:] != x.shape[2:]:
            grad_reference = F.interpolate(grad_reference, size=x.shape[2:], mode='bilinear', align_corners=False)
        grad_x = self.compute_gradient(x)
        grad_mask = (grad_reference > grad_reference.mean() * 1.5).float()
        grad_diff = grad_reference - grad_x
        grad_correction = grad_diff * grad_mask * weight
        x_structure = x + grad_correction
        return x_structure

    def apply_physical_consistency(self, x, x_ir, x_vis, weight=0.1):
        if x_ir.shape[2:] != x.shape[2:]:
            x_ir = F.interpolate(x_ir, size=x.shape[2:], mode='bilinear', align_corners=False)
        if x_vis.shape[2:] != x.shape[2:]:
            x_vis = F.interpolate(x_vis, size=x.shape[2:], mode='bilinear', align_corners=False)
        heat_info = F.avg_pool2d(x_ir, kernel_size=3, stride=1, padding=1)
        padded_vis = F.pad(x_vis, (1, 1, 1, 1), mode='replicate')
        laplacian_vis = (
            padded_vis[:, :, :-2, 1:-1] + 
            padded_vis[:, :, 2:, 1:-1] + 
            padded_vis[:, :, 1:-1, :-2] + 
            padded_vis[:, :, 1:-1, 2:] - 
            4 * x_vis
        ).abs()
        heat_mask = (heat_info > heat_info.mean()).float()
        structure_mask = (laplacian_vis > laplacian_vis.mean()).float()
        x_physics = (
            x * (1 - weight) + 
            x_ir * heat_mask * weight * 0.6 + 
            x_vis * structure_mask * weight * 0.4
        )
        return x_physics

    def forward(self, inputs):
        data_dict = {}

        b = self.betas.to(inputs.device)

        if self.training:
            x_vis = inputs[:, :1, :, :]  
            x_ir = inputs[:, 1:, :, :] 

            x_ir_aligned = optimal_transport_alignment(x_ir, x_vis)
            inputs_ot = torch.cat([x_ir_aligned, x_vis], dim=1)
            output = self.encoder(inputs_ot, pred_fea=None)
            z, mu, logvar, features_down = output["z"], output["mu"], output["logvar"], output["features"]
            sigma = torch.exp(0.5 * logvar)
            mu_map     = mu.expand_as(z)
            sigma_map  = sigma.expand_as(z)
            z_sample   = mu_map + sigma_map * torch.randn_like(z)
            z_norm = utils.data_transform(z_sample)
            t = torch.randint(low=0, high=self.num_timesteps, size=(z_norm.shape[0] // 2 + 1,)).to(self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:z_norm.shape[0]].to(inputs.device)
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            e = torch.randn_like(z_norm)
            x = z_norm * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([z_norm, x], dim=1), t.float())
            pred_fea = self.physics_guided_sample(z_norm, b, x_ir_aligned, x_vis)
            pred_fea = utils.inverse_data_transform(pred_fea)
            pred_x = self.encoder(inputs, pred_fea=pred_fea, features_down=features_down)["pred_img"]
            data_dict["pred_x"] = pred_x
            data_dict["noise_output"] = noise_output
            data_dict["e"] = e
            data_dict["pred_fea"] = pred_fea
            data_dict["z_norm"] = z_norm
            data_dict["mu"] = mu
            data_dict["logvar"] = logvar
        else:
            output = self.encoder(inputs, pred_fea=None)
            z, mu, logvar, features_down = output["z"], output["mu"], output["logvar"], output["features"]
            features_down = output["features"]
            sigma    = torch.exp(0.5 * logvar)
            mu_map     = mu.expand_as(z)
            sigma_map  = sigma.expand_as(z)
            z_sample   = mu_map + sigma_map * torch.randn_like(z)
            z_norm = utils.data_transform(z_sample)
            x_vis = inputs[:, :1, :, :] 
            x_ir = inputs[:, 1:, :, :]
            pred_fea = self.physics_guided_sample(z_norm, b, x_ir, x_vis)
            pred_fea = utils.inverse_data_transform(pred_fea)
            pred_x = self.encoder(inputs, pred_fea=pred_fea, features_down=features_down)["pred_img"]
            data_dict["pred_x"] = pred_x
        return data_dict
def compute_gradient(image):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(image.shape[1], 1, 1, 1)  # [C, 1, 3, 3]
    sobel_y = sobel_y.repeat(image.shape[1], 1, 1, 1)  # [C, 1, 3, 3]
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.shape[1])
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.shape[1])
    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient
def fusion_loss(pred_fuse, ir, vis, alpha5=1.0, alpha6=1.0, alpha7=1.0):
    intensity_loss = alpha5 * F.mse_loss(pred_fuse, ir) + alpha6 * F.mse_loss(pred_fuse, vis)
    grad_fuse = compute_gradient(pred_fuse)
    grad_ir = compute_gradient(ir)
    grad_vis = compute_gradient(vis)
    max_grad = torch.max(grad_ir, grad_vis)
    gradient_loss = alpha7 * F.mse_loss(grad_fuse, max_grad)
    total_loss = intensity_loss + gradient_loss
    return total_loss

class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.start_epoch, self.step = checkpoint['epoch'], checkpoint['step']
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                output = self.model(x)

                noise_loss, ib_loss, fusion_loss = self.noise_estimation_loss(output, x[:, :1, :, :], x[:, 1:, :, :])
                loss = 100*noise_loss + 10*ib_loss + 10*fusion_loss

                data_time += time.time() - data_start

                if self.step % 10 == 0:
                    print("step:{}, total_loss:{:.5f} noise_loss:{:.5f} ib_loss:{:.5f} fusion_loss:{:.5f}  time:{:.5f}".
                        format(self.step, loss.item(), 100*noise_loss.item(), 
                                10*ib_loss.item(), 10*fusion_loss.item(), data_time / (i + 1)))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)
                    checkname=f"checkpoint_epoch_{epoch}_{self.step}"
                    utils.logging.save_checkpoint({'step': self.step,
                                                'epoch': epoch + 1,
                                                'state_dict': self.model.state_dict(),
                                                'optimizer': self.optimizer.state_dict(),
                                                'ema_helper': self.ema_helper.state_dict(),
                                                'params': self.args,
                                                'config': self.config},
                                                filename=os.path.join(self.config.data.ckpt_dir, checkname))


    def noise_estimation_loss(self, output, vi, ir):
        mu, logvar, pred_x = output["mu"], output["logvar"], output["pred_x"]
        noise_output, e = output["noise_output"], output["e"]
        noise_loss = F.smooth_l1_loss(noise_output, e)
        ib_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        fusion_losses = fusion_loss(pred_x, ir, vi)
        return noise_loss, ib_loss, fusion_losses


    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder,
                                    self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()

        with torch.no_grad():
            print('Performing validation at step: {}'.format(step))
            for i, (x, y,vi_cb,vi_cr) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape

                img_h_64 = int(64 * np.ceil(img_h / 64.0))
                img_w_64 = int(64 * np.ceil(img_w / 64.0))
                x = F.pad(x, (0, img_w_64 - img_w, 0, img_h_64 - img_h), 'reflect')
                pred_x = self.model(x.to(self.device))["pred_x"][:, :, :img_h, :img_w]
                pred_rgb = YCbCr2RGB(pred_x, vi_cb.to(self.device), vi_cr.to(self.device))
                utils.logging.save_image(pred_rgb, os.path.join(image_folder, str(step), '{}'.format(y[0])))
