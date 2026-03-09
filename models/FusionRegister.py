import torchvision.transforms as T
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2
from models.layers import *
from models.covmlp import *
import random
import torchvision.transforms.functional as TF


def ma(mask, percent, edge):
    B, _, H, W = mask.shape

    binary_mask = torch.zeros_like(mask, dtype=torch.float32)
    if H <= 2 or W <= 2:
        return binary_mask

    non_edge_mask = mask[:, :, edge:-edge, edge:-edge]
    non_edge_values = non_edge_mask.flatten(start_dim=1)

    for b in range(B):
        values = non_edge_values[b]
        N = values.numel()

        if N == 0:
            continue

        sorted_values, _ = torch.sort(values)

        top_k = int(N * percent)
        bottom_k = int(N * percent)

        if top_k == 0 or bottom_k == 0:
            continue

        top_threshold = sorted_values[-top_k]
        bottom_threshold = sorted_values[bottom_k - 1]

        non_edge_binary = (non_edge_mask[b] >= top_threshold) | (
                    non_edge_mask[b] <= bottom_threshold)
        non_edge_binary = non_edge_binary.float()

        binary_mask[b, :, edge:-edge, edge:-edge] = non_edge_binary

    return binary_mask.cuda()

class fusionnet(nn.Module):
    def __init__(self):
        super(fusionnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=0),
            )
        self.act1 = nn.Sequential(nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=0),
            )
        self.act2= nn.Sequential(nn.Tanh())
    def forward(self, input):
        G11 = self.conv1(input)
        G11_1 = self.act1(G11)
        G21 = self.conv2(G11_1)
        G21_1 = self.act2(G21)

        return G21_1

def filter_and_connect_mask(mask_tensor, min_area=100, morph_kernel=9):

    B, _, H, W = mask_tensor.shape
    output = []

    for i in range(B):
        mask_np = mask_tensor[i, 0].cpu().numpy().astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        filtered = np.zeros_like(closed)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered[labels == label] = 255

        filtered_tensor = torch.from_numpy(filtered.astype(np.float32) / 255.0).unsqueeze(0)
        output.append(filtered_tensor)
    return torch.stack(output, dim=0).cuda()

def flow_to_color(flow, sigma=1.0, magnitude_threshold=0.01, value_power=0.5):

    flow_x = flow[0, :, :]
    flow_y = flow[1, :, :]

    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
    angle = np.arctan2(flow_y, flow_x)


    mask = magnitude < magnitude_threshold
    magnitude[mask] = 0


    max_magnitude = np.max(magnitude)
    if max_magnitude > 1e-5:
        magnitude = magnitude / max_magnitude
    magnitude = np.clip(magnitude, 0, 1)

    magnitude_adjusted = np.power(magnitude, value_power)

    hue = ((angle + np.pi) / (2 * np.pi)) * 360
    saturation = magnitude_adjusted
    value = np.ones_like(magnitude)

    hsv = np.stack([hue, saturation * 255, value * 255], axis=-1).astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    rgb[mask] = [255, 255, 255]

    return rgb

def save_tensor_as_image(tensor, i, name, fea=False):
    output_dir = './results/epoch_'+str(i)+'/'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{name}.png")
    if fea:
        if tensor.shape[1] == 2:
            flow = tensor[0].detach().cpu().numpy()
            rgb_image = flow_to_color(flow)

            H, W = tensor.shape[2], tensor.shape[3]
            figsize = (W / 100, H / 100)

            plt.figure(figsize=figsize)
            plt.imshow(rgb_image)
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            fu1 = torch.mean(tensor, dim=1)
            image = (fu1).squeeze().detach().cpu().numpy()
            plt.imshow(image, cmap='jet')
            plt.axis('off')
            plt.savefig(filename)
            plt.close()
    else:
        normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        save_image(normalized_tensor, filename)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.body(x)

def random_light_transform(vi, ir, vi_ori, ir_ori, cb, cr, cls):
    B, C, H_in, W_in = vi.shape  # e.g., (B, 1, 260, 260)

    # Output lists
    vi_output_list = []
    ir_output_list = []
    mask_output_list = []

    final_vi_ori_list = []
    final_ir_ori_list = []
    cb_list = []
    cr_list = []

    blur_transform = T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
    target_H, target_W = 256, 256
    crop_h_start = (H_in - target_H) // 2
    crop_w_start = (W_in - target_W) // 2

    for i in range(B):
        item_vi = vi[i].clone()
        item_ir = ir[i].clone()
        item_vi_ori = vi_ori[i].clone()
        item_ir_ori = ir_ori[i].clone()
        item_cb = cb[i].clone()
        item_cr = cr[i].clone()

        current_final_vi_output = None
        current_final_ir_output = None
        current_final_mask_output = None

        if cls[i] == 0:
            current_mask = torch.ones_like(item_vi)

            angle = random.uniform(-2, 2)
            trans_x = random.uniform(-2, 2)
            trans_y = random.uniform(-2, 2)
            scale = random.uniform(0.95, 1.08)

            transformed_ir = TF.affine(
                item_ir, angle=angle, translate=(trans_x, trans_y), scale=scale, shear=0,
                interpolation=TF.InterpolationMode.BILINEAR
            )
            transformed_mask = TF.affine(
                current_mask, angle=angle, translate=(trans_x, trans_y), scale=scale, shear=0,
                interpolation=TF.InterpolationMode.NEAREST
            )
            transformed_vi = item_vi

            if random.random() < 0.5:
                transform_choice = random.choice(['hflip', 'vflip', 'rot90'])
                tensors_to_transform_cls0 = [transformed_vi, transformed_ir, transformed_mask,
                                             item_vi_ori, item_ir_ori, item_cb, item_cr]

                transformed_list_cls0 = []

                if transform_choice == 'hflip':
                    for tensor_item in tensors_to_transform_cls0:
                        transformed_list_cls0.append(TF.hflip(tensor_item))
                elif transform_choice == 'vflip':
                    for tensor_item in tensors_to_transform_cls0:
                        transformed_list_cls0.append(TF.vflip(tensor_item))
                elif transform_choice == 'rot90':
                    k_rot = random.choice([1, 2, 3])
                    for tensor_item in tensors_to_transform_cls0:
                        transformed_list_cls0.append(torch.rot90(tensor_item, k_rot, dims=[1, 2]))

                if transformed_list_cls0:  # If a transform was applied
                    transformed_vi, transformed_ir, transformed_mask, \
                    item_vi_ori, item_ir_ori, item_cb, item_cr = transformed_list_cls0

            if H_in >= target_H and W_in >= target_W:
                current_final_vi_output = transformed_vi[:, crop_h_start:crop_h_start + target_H,
                                          crop_w_start:crop_w_start + target_W]
                current_final_ir_output = transformed_ir[:, crop_h_start:crop_h_start + target_H,
                                          crop_w_start:crop_w_start + target_W]
                current_final_mask_output = transformed_mask[:, crop_h_start:crop_h_start + target_H,
                                            crop_w_start:crop_w_start + target_W]
            else:
                current_final_vi_output = transformed_vi

                current_final_ir_output = transformed_ir

                current_final_mask_output = transformed_mask
            current_final_mask_output = (current_final_mask_output >= 0.5).float()
            shrinkage = random.randint(2, 3)
            if current_final_mask_output.shape[1] > 2 * shrinkage and current_final_mask_output.shape[
                2] > 2 * shrinkage:
                current_final_mask_output[:, :shrinkage, :] = 0
                current_final_mask_output[:, -shrinkage:, :] = 0
                current_final_mask_output[:, :, :shrinkage] = 0
                current_final_mask_output[:, :, -shrinkage:] = 0

            final_vi_ori_list.append(item_vi_ori)  # Appends (H_in, W_in)
            final_ir_ori_list.append(item_ir_ori)  # Appends (H_in, W_in)

        elif cls[i] == 1:
            k_rot = random.choice([1, 2, 3])

            vi_rotated = torch.rot90(item_vi, k_rot, dims=[1, 2])
            ir_rotated = torch.rot90(item_ir, k_rot, dims=[1, 2])

            item_vi_ori_rotated = torch.rot90(item_vi_ori, k_rot, dims=[1, 2])
            item_ir_ori_rotated = torch.rot90(item_ir_ori, k_rot, dims=[1, 2])
            item_cb = torch.rot90(item_cb, k_rot, dims=[1, 2])
            item_cr = torch.rot90(item_cr, k_rot, dims=[1, 2])

            current_mask_cls1 = torch.ones_like(item_vi)
            mask_rotated_cls1 = torch.rot90(current_mask_cls1, k_rot, dims=[1, 2])

            ir_blurred_rotated = blur_transform(ir_rotated)

            if H_in >= target_H and W_in >= target_W:
                current_final_vi_output = vi_rotated[:, crop_h_start:crop_h_start + target_H,
                                          crop_w_start:crop_w_start + target_W]
                current_final_ir_output = ir_blurred_rotated[:, crop_h_start:crop_h_start + target_H,
                                          crop_w_start:crop_w_start + target_W]
                current_final_mask_output = mask_rotated_cls1[:, crop_h_start:crop_h_start + target_H,
                                            crop_w_start:crop_w_start + target_W]
            else:
                current_final_vi_output = vi_rotated

                current_final_ir_output = ir_blurred_rotated
                current_final_mask_output = mask_rotated_cls1

            final_vi_ori_list.append(item_vi_ori_rotated)
            final_ir_ori_list.append(item_ir_ori_rotated)

        vi_output_list.append(current_final_vi_output)
        ir_output_list.append(current_final_ir_output)
        mask_output_list.append(current_final_mask_output)

        cb_list.append(item_cb)
        cr_list.append(item_cr)

    return torch.stack(vi_output_list), torch.stack(ir_output_list), \
            torch.stack(final_vi_ori_list), torch.stack(final_ir_ori_list), \
            torch.stack(cb_list), torch.stack(cr_list), torch.stack(mask_output_list)

def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out

def RGB2YCrCb(rgb_image):

    rgb_image=rgb_image
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0).detach()
    Cb = Cb.clamp(0.0, 1.0).detach()
    Cb=Cb.squeeze(0)
    Cr=Cr.squeeze(0)
    return Y, Cb, Cr

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=ResBlock):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane//2, kernel_size=3, stride=1, relu=True),
            # BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-inchannel, kernel_size=1, stride=1, relu=True)
        )
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)
    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)
    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()

    device = flow.device

    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype),
            indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.type(x.type())
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output

class FusionRegister(nn.Module):
    def __init__(self,
                inp_channels=3,
                out_channels=3,
                dim=16,
                num_blocks=[12,12],
                kernel_size=7,
                inference=False,
                ):
        super(FusionRegister, self).__init__()
        self.inference = inference
        self.dim = dim
        self.kernel_size = kernel_size
        self.kernel_pad = int((self.kernel_size - 1) / 2.0)
        if not inference:
            BasicConv = BasicConv_do
            DwConv = lambda *args, **kwargs: Conv_dw(*args, simam=False, **kwargs)
        else:
            BasicConv = BasicConv_do_eval
            DwConv = lambda *args, **kwargs: Conv_dw(*args, simam=False, **kwargs)
        base_channel = dim

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_blocks[0]),
            EBlock(base_channel * 2, num_blocks[0]),

            EBlock(base_channel, num_blocks[1]),
            EBlock(base_channel, num_blocks[1]),

        ])

        self.feat_extract = nn.ModuleList([
            DwConv(inp_channels, base_channel, kernel_size=3, relu=True, stride=1),
            DwConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            DwConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1),
            DwConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            DwConv(1, base_channel, kernel_size=3, relu=True, stride=1),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 2, num_blocks[0]),
            DBlock(base_channel, num_blocks[0])
        ])

        self.Convs = nn.ModuleList([
            DwConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 3, base_channel*1, BasicConv=BasicConv),
            AFF(base_channel * 3, base_channel*2, BasicConv=BasicConv)
        ])

        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

        self.softmax = nn.Softmax(1)

        self.att_v = CMWMLP_block(base_channel, base_channel, use_corr=True, att="chan")
        self.att_i = CMWMLP_block(base_channel, base_channel, use_corr=True, att="space")

        self.KernelPredictFlow = nn.ModuleList([
                DwConv(base_channel * 2, 2, kernel_size=3, relu=False, stride=1),
                DwConv(base_channel, 2, kernel_size=3, relu=False, stride=1),
        ])
        self.flowup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.KernelPredictFlowMask = nn.ModuleList([

                DwConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
                DwConv(base_channel, 1, kernel_size=3, relu=False, stride=1),
        ])
        self.sigmoid = nn.Sigmoid()
        self.KernelOutBias = nn.ModuleList([

                DwConv(base_channel, out_channels, kernel_size=3, relu=False, stride=1),
        ])

        self.up = Upsample(base_channel*2)

    def forward(self, x, vi, ir):
        # print(x.shape,vi.shape,ir.shape)
        x_2 = F.interpolate(x, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        mask = []
        flow = []
        outputs_fil = list()
        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)

        res2 = self.AFFs[1](z12, res2)
        res1 = self.AFFs[0](res1, z21)

        vif = self.Encoder[2](self.feat_extract[3](vi))
        irf = self.Encoder[3](self.feat_extract[4](ir))
        z = self.Decoder[0](res2)
        s2_kernal_flow = self.KernelPredictFlow[0](z)
        s2_kernal_flowmask = self.KernelPredictFlowMask[0](z)
        s2_kernal_flowmask = self.sigmoid(s2_kernal_flowmask)
        flow.append(s2_kernal_flow)
        zx2 = torch.cat([z, x_2], 1)
        s2_kernal_flowfeat0, x_2_0 = torch.split(flow_warp(zx2, s2_kernal_flow.permute(0, 2, 3, 1)), self.dim * 2,
                                                 dim=1)
        s2_kernal_flowfeat1, x_2_1 = torch.split(flow_warp(zx2, -s2_kernal_flow.permute(0, 2, 3, 1)), self.dim * 2,
                                                 dim=1)

        if not self.inference:
            x_2 = x_2_0 * s2_kernal_flowmask + x_2_1 * (1 - s2_kernal_flowmask)
            mask.append(s2_kernal_flowmask)
            outputs_fil.append(x_2)

        z = torch.cat(
            [z, s2_kernal_flowfeat0 * s2_kernal_flowmask + s2_kernal_flowfeat1 * (1 - s2_kernal_flowmask)], 1)

        z = self.feat_extract[2](z)
        z = self.up(z)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        s1_kernal_flow = self.KernelPredictFlow[1](z) + self.flowup(s2_kernal_flow) * 2
        s1_kernal_flowmask = self.KernelPredictFlowMask[1](z)
        s1_kernal_flowmask = self.sigmoid(s1_kernal_flowmask)

        mask.append(s1_kernal_flowmask)
        zx = torch.cat([z, x], 1)
        s1_kernal_flowfeat0, x_0 = torch.split(flow_warp(zx, s1_kernal_flow.permute(0, 2, 3, 1)), self.dim, dim=1)
        s1_kernal_flowfeat1, x_1 = torch.split(flow_warp(zx, -s1_kernal_flow.permute(0, 2, 3, 1)), self.dim, dim=1)
        x = x_0 * s1_kernal_flowmask + x_1 * (1 - s1_kernal_flowmask)

        flow.append(s1_kernal_flow)

        z_1 = z

        fea = s1_kernal_flowfeat0 * s1_kernal_flowmask + s1_kernal_flowfeat1 * (1 - s1_kernal_flowmask)
        z_v = self.att_v(fea, vif)
        z_i = self.att_i(fea, irf)
        z = z_i + z_v + fea

        z_6 = z

        s1_kernal_bias = self.KernelOutBias[0](z)

        out = (x - x.max()) / (x.max() - x.min())
        nobias = out
        out += s1_kernal_bias
        if not self.inference:
            outputs.append(out)
            outputs_fil.append(x)
            return out, outputs_fil[::-1], mask, flow, s1_kernal_bias, nobias, [z_1, z_v, z_i, z_6]
        else:
            return out

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

def calculate_parameters(model):
    total_params = 0
    total_size = 0

    for param in model.parameters():
        param_count = param.numel()
        param_size = param.element_size() * param_count

        total_params += param_count
        total_size += param_size

    return total_params, total_size


if __name__ == "__main__":
    model = FusionRegister().cuda(1)
    x = torch.randn(1, 3,  480, 640).cuda(1)
    z = torch.randn(1, 1,  480, 640).cuda(1)
    with torch.no_grad():
        y,_,_,_,_,_,_ = model(x,x,z)
    num_params, storage_size = calculate_parameters(model)
    storage_size_mb = storage_size / (1024 ** 2)  # 1MB = 1024*1024 Bytes

    print(f'总参数数量: {num_params}')
    print(f'模型存储大小: {storage_size_mb:.2f} MB')

    from thop import profile
    flops, params = profile(model, inputs=(x, x, z))
    gflops = flops / 1e9
    print(f'GFLOPs: {gflops:.4f}')