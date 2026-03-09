import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)


class Gradloss(nn.Module):
    def __init__(self):
        super(Gradloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss()

    def forward(self,  image_vis, generate_img):
        y_grad = self.sobelconv(image_vis)
        generate_img_grad = self.sobelconv(generate_img)
        loss_grad = self.mse_criterion(generate_img_grad, y_grad)
        return loss_grad

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss

class TextureLoss(nn.Module):
    def __init__(self, kernel_size=7, eps=1e-6,
                 lambda_vis=1.0, lambda_ir=1.0):

        super().__init__()
        self.ks = kernel_size
        self.eps = eps
        self.lambda_vis = lambda_vis
        self.lambda_ir  = lambda_ir
        self.pad = kernel_size // 2

        # 均值卷积核 1×1×ks×ks
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size**2)
        self.register_buffer('kernel', kernel)
        self.l1 = CharbonnierLoss()

    def _local_variance(self, x):
        # 保证kernel在同一个device上
        kern = self.kernel.to(x.device).repeat(x.shape[1], 1, 1, 1)  # (C,1,ks,ks)
        x_pad   = F.pad(x,    (self.pad,)*4, mode='reflect')
        x2_pad  = F.pad(x*x,  (self.pad,)*4, mode='reflect')
        mean    = F.conv2d(x_pad,   kern, groups=x.shape[1])
        mean2   = F.conv2d(x2_pad,  kern, groups=x.shape[1])
        var = mean2 - mean*mean
        return var.clamp(min=self.eps)

    def forward(self,  I_vis, F):
        var_F_vis = self._local_variance(F)           # (B,3,H,W)
        var_vis   = self._local_variance(I_vis)       # (B,3,H,W)
        loss_vis  = self.l1(var_F_vis, var_vis)
        return loss_vis