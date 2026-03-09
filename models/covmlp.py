import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal


class CorrMLP(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 enc_channels: int = 8,
                 dec_channels: int = 16,
                 use_checkpoint: bool = True):
        super().__init__()

        self.Encoder = Conv_encoder_2D(in_channels=in_channels,
                                       channel_num=enc_channels,
                                       use_checkpoint=use_checkpoint)
        self.Decoder = MLP_decoder_2D(in_channels_encoder=enc_channels,
                                      channel_num_decoder=dec_channels,
                                      use_checkpoint=use_checkpoint)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, fixed, moving):
        x_fix = self.Encoder(fixed)
        x_mov = self.Encoder(moving)
        flow = self.Decoder(x_fix, x_mov)
        warped = self.SpatialTransformer(moving, flow)

        return warped, flow


class Conv_encoder_2D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.Convblock_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.Convblock_2 = Conv_block(channel_num, channel_num * 2, use_checkpoint)
        self.Convblock_3 = Conv_block(channel_num * 2, channel_num * 4, use_checkpoint)
        self.Convblock_4 = Conv_block(channel_num * 4, channel_num * 8, use_checkpoint)
        self.downsample = nn.AvgPool2d(2, stride=2)

    def forward(self, x_in):
        x_1 = self.Convblock_1(x_in)
        x = self.downsample(x_1)
        x_2 = self.Convblock_2(x)
        x = self.downsample(x_2)
        x_3 = self.Convblock_3(x)
        x = self.downsample(x_3)
        x_4 = self.Convblock_4(x)

        return [x_1, x_2, x_3, x_4]


class MLP_decoder_2D(nn.Module):

    def __init__(self,
                 in_channels_encoder: int,
                 channel_num_decoder: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.mlp_11 = CMWMLP_block(in_channels_encoder, channel_num_decoder, use_corr=True)
        self.mlp_12 = CMWMLP_block(in_channels_encoder * 2, channel_num_decoder * 2, use_corr=True)
        self.mlp_13 = CMWMLP_block(in_channels_encoder * 4, channel_num_decoder * 4, use_corr=True)
        self.mlp_14 = CMWMLP_block(in_channels_encoder * 8, channel_num_decoder * 8, use_corr=True)

        self.mlp_21 = CMWMLP_block(channel_num_decoder, channel_num_decoder, use_corr=True)
        self.mlp_22 = CMWMLP_block(channel_num_decoder * 2, channel_num_decoder * 2, use_corr=True)
        self.mlp_23 = CMWMLP_block(channel_num_decoder * 4, channel_num_decoder * 4, use_corr=True)

        self.upsample_1 = PatchExpanding_block(embed_dim=channel_num_decoder * 2)
        self.upsample_2 = PatchExpanding_block(embed_dim=channel_num_decoder * 4)
        self.upsample_3 = PatchExpanding_block(embed_dim=channel_num_decoder * 8)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2,
                                                         mode='bilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

        self.reghead_1 = RegHead_block(channel_num_decoder, use_checkpoint)
        self.reghead_2 = RegHead_block(channel_num_decoder * 2, use_checkpoint)
        self.reghead_3 = RegHead_block(channel_num_decoder * 4, use_checkpoint)
        self.reghead_4 = RegHead_block(channel_num_decoder * 8, use_checkpoint)

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3, x_fix_4 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4 = x_mov

        x_4 = self.mlp_14(x_fix_4, x_mov_4)
        flow_4 = self.reghead_4(x_4)

        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3_warped = self.SpatialTransformer(x_mov_3, flow_4_up)

        x = self.mlp_13(x_fix_3, x_mov_3_warped)
        x_3 = self.mlp_23(x, self.upsample_3(x_4))

        x = self.reghead_3(x_3)
        flow_3 = x + flow_4_up

        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2_warped = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.mlp_12(x_fix_2, x_mov_2_warped)
        x_2 = self.mlp_22(x, self.upsample_2(x_3))

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1_warped = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.mlp_11(x_fix_1, x_mov_1_warped)
        x_1 = self.mlp_21(x, self.upsample_1(x_2))

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


class SpatialTransformer_block(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s, device=src.device) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type_as(flow)

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='bilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x


class Conv_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.Conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_1 = nn.InstanceNorm2d(out_channels)

        self.Conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_2 = nn.InstanceNorm2d(out_channels)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def Conv_forward(self, x_in):
        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)

        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)

        return x_out

    def forward(self, x_in):
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in, use_reentrant=False)
        else:
            x_out = self.Conv_forward(x_in)
        return x_out


class RegHead_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.reg_head = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding='same')
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x_in):
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.reg_head, x_in, use_reentrant=False)
        else:
            x_out = self.reg_head(x_in)
        return x_out


class PatchExpanding_block(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim // 2)

    def forward(self, x_in):
        x = self.up_conv(x_in)
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x_out = einops.rearrange(x, 'b h w c -> b c h w')
        return x_out


class CMWMLP_block(nn.Module):

    def __init__(self, in_channels, out_channels, use_corr=True, corr_max_disp=1, att="space"):
        super().__init__()
        self.use_corr = use_corr
        self.corr_max_disp = corr_max_disp

        if use_corr:
            self.Corr = Correlation(max_disp=self.corr_max_disp)
            correlation_channels = (2 * self.corr_max_disp + 1) ** 2
            self.Conv = nn.Conv2d(in_channels * 2 + correlation_channels, out_channels, kernel_size=3,
                                  stride=1, padding='same')
        else:
            self.Conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding='same')

        self.mlpLayer = MultiWinMlpLayer(out_channels)
        if att == "space":
            self.attention = RSAB(out_channels)
        elif att == "chan":
            self.attention = RCAB(out_channels)

    def forward(self, x_1, x_2):
        x_corr = self.Corr(x_1, x_2)
        x = torch.cat([x_1, x_corr, x_2], dim=1)
        x = self.Conv(x)
        shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.mlpLayer(x)
        x = self.attention(x)
        x = x.permute(0, 3, 1, 2)
        x_out = x + shortcut
        return x_out

class MultiWinMlpLayer(nn.Module):

    def __init__(self, num_channels, use_bias=True):
        super().__init__()
        self.num_channels = num_channels

        self.WinGmlpLayer_1 = WinGmlpLayer(win_size=[1, 1], num_channels=num_channels, use_bias=use_bias)
        self.WinGmlpLayer_2 = WinGmlpLayer(win_size=[3, 3], num_channels=num_channels, use_bias=use_bias)
        self.WinGmlpLayer_3 = WinGmlpLayer(win_size=[5, 5], num_channels=num_channels, use_bias=use_bias)


        self.reweight = MLP(num_channels, num_channels // 4, num_channels * 3)
        self.out_project = nn.Linear(num_channels, num_channels, bias=use_bias)

    def forward(self, x_in):
        n, h, w, c_num = x_in.shape
        x_1 = self.WinGmlpLayer_1(x_in)
        x_2 = self.WinGmlpLayer_2(x_in)
        x_3 = self.WinGmlpLayer_3(x_in)
        _a_sum = (x_1 + x_2 + x_3).permute(0, 3, 1, 2)
        _a_pooled = _a_sum.flatten(2).mean(dim=2)
        _a_reweighted = self.reweight(_a_pooled)
        _a_reshaped = _a_reweighted.reshape(n, self.num_channels, 3)
        a_coefficients = _a_reshaped.permute(2, 0, 1)
        a_coefficients = a_coefficients.softmax(dim=0)
        factor_0 = a_coefficients[0].unsqueeze(1).unsqueeze(1)
        factor_1 = a_coefficients[1].unsqueeze(1).unsqueeze(1)
        factor_2 = a_coefficients[2].unsqueeze(1).unsqueeze(1)
        x = x_1 * factor_0 + \
            x_2 * factor_1 + x_3 * factor_2
        x = self.out_project(x)
        x_out = x + x_in
        return x_out


# class MultiWinMlpLayer(nn.Module):
#
#     def __init__(self, num_channels, use_bias=True):
#         super().__init__()
#         self.num_channels = num_channels
#         self.WinGmlpLayer_1 = WinGmlpLayer(win_size=[3, 3], num_channels=num_channels, use_bias=use_bias)
#         self.WinGmlpLayer_2 = WinGmlpLayer(win_size=[5, 5], num_channels=num_channels, use_bias=use_bias)
#
#         self.reweight = MLP(num_channels, num_channels // 4, num_channels * 2)
#         self.out_project = nn.Linear(num_channels, num_channels, bias=use_bias)
#
#     def forward(self, x_in):
#         n, h, w, c_num = x_in.shape
#         x_1 = self.WinGmlpLayer_1(x_in)
#         x_2 = self.WinGmlpLayer_2(x_in)
#         _a_sum = (x_1 + x_2).permute(0, 3, 1, 2)
#         _a_pooled = _a_sum.flatten(2).mean(dim=2)
#         _a_reweighted = self.reweight(_a_pooled)
#         _a_reshaped = _a_reweighted.reshape(n, self.num_channels, 2)
#         a_coefficients = _a_reshaped.permute(2, 0, 1)
#         a_coefficients = a_coefficients.softmax(dim=0)
#         factor_0 = a_coefficients[0].unsqueeze(1).unsqueeze(1)
#         factor_1 = a_coefficients[1].unsqueeze(1).unsqueeze(1)
#         x = x_1 * factor_0 + \
#             x_2 * factor_1
#         x = self.out_project(x)
#         x_out = x + x_in
#         return x_out


class WinGmlpLayer(nn.Module):

    def __init__(self, win_size, num_channels, factor=2, use_bias=True):
        super().__init__()

        self.fh = win_size[0]
        self.fw = win_size[1]
        self.in_project = nn.Linear(num_channels, num_channels * factor, bias=use_bias)
        self.gelu = nn.GELU()
        self.SpatialGatingUnit = SpatialGatingUnit(num_channels * factor,
                                                   n=self.fh * self.fw)
        self.out_project = nn.Linear(num_channels * factor // 2, num_channels, bias=use_bias)

    def forward(self, x):
        n, h, w, c = x.shape

        pad_h_bottom = (self.fh - h % self.fh) % self.fh
        pad_w_right = (self.fw - w % self.fw) % self.fw

        x_padded = nnf.pad(x, (0, 0, 0, pad_w_right, 0, pad_h_bottom))

        padded_h, padded_w = x_padded.shape[1], x_padded.shape[2]
        gh = padded_h // self.fh
        gw = padded_w // self.fw
        x_split = split_images(x_padded, patch_size=(self.fh, self.fw))
        shortcut = x_split
        x_gmlp = self.in_project(x_split)
        x_gmlp = self.gelu(x_gmlp)
        x_gmlp = self.SpatialGatingUnit(x_gmlp)
        x_gmlp = self.out_project(x_gmlp)
        x_gmlp = x_gmlp + shortcut

        x_unsplit = unsplit_images(x_gmlp, grid_size=(gh, gw), patch_size=(self.fh, self.fw))

        if pad_h_bottom > 0 or pad_w_right > 0:
            x_out = x_unsplit[:, :h, :w, :].contiguous()
        else:
            x_out = x_unsplit.contiguous()

        return x_out


class SpatialGatingUnit(nn.Module):

    def __init__(self, c_factor, n, use_bias=True):
        super().__init__()
        self.c_half = c_factor // 2
        self.Dense_0 = nn.Linear(n, n, bias=use_bias)

    def forward(self, x):
        u, v = torch.split(x, self.c_half, dim=-1)
        v = v.permute(0, 1, 3, 2)
        v = self.Dense_0(v)
        v = v.permute(0, 1, 3, 2)

        return u * (v + 1.0)


class RSAB(nn.Module):

    def __init__(self, num_channels, lrelu_slope=0.2, use_bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.leaky_relu = nn.LeakyReLU(negative_slope=lrelu_slope)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.spatial_attention = SpatialAttention()

    def forward(self, x_in_hwc):
        shortcut = x_in_hwc

        x = x_in_hwc.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)

        x = self.spatial_attention(x)
        x_out = x + shortcut
        return x_out


class SpatialAttention(nn.Module):

    def __init__(self, use_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=use_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in_hwc):
        max_out, _ = torch.max(x_in_hwc, dim=3, keepdim=True)
        avg_out = torch.mean(x_in_hwc, dim=3, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=3)

        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x.permute(0, 2, 3, 1)

        x_out = x_in_hwc * x
        return x_out


class RCAB(nn.Module):

    def __init__(self, num_channels, reduction=4, lrelu_slope=0.2, use_bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.leaky_relu = nn.LeakyReLU(negative_slope=lrelu_slope)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.channel_attention = CALayer(num_channels=num_channels, reduction=reduction)

    def forward(self, x_in_hwc):
        shortcut = x_in_hwc

        x = x_in_hwc.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)

        x = self.channel_attention(x)
        x_out = x + shortcut
        return x_out


class CALayer(nn.Module):

    def __init__(self, num_channels, reduction=4, use_bias=True):
        super().__init__()

        self.Conv_0 = nn.Conv2d(num_channels, num_channels // reduction, kernel_size=1, stride=1, bias=use_bias)
        self.relu = nn.ReLU()
        self.Conv_1 = nn.Conv2d(num_channels // reduction, num_channels, kernel_size=1, stride=1, bias=use_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in_hwc):
        x = x_in_hwc.permute(0, 3, 1, 2)
        x_squeezed = torch.mean(x, dim=(2, 3), keepdim=True)

        x_excited = self.Conv_0(x_squeezed)
        x_excited = self.relu(x_excited)
        x_excited = self.Conv_1(x_excited)
        w = self.sigmoid(x_excited)
        x_out = x_in_hwc * w.permute(0, 2, 3, 1)
        return x_out


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Correlation(nn.Module):

    def __init__(self, max_disp=1, kernel_size=1, stride=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.max_disp = max_disp
        self.padlayer = nn.ConstantPad2d(max_disp, 0)

    def forward(self, x_1, x_2):
        x_2_padded = self.padlayer(x_2)
        disp_range = torch.arange(-self.max_disp, self.max_disp + 1, device=x_1.device)
        offsety, offsetx = torch.meshgrid(disp_range, disp_range, indexing='ij')
        b, c, h, w = x_1.shape
        correlations = []
        for dy, dx in zip(offsety.reshape(-1), offsetx.reshape(-1)):
            y_start = self.max_disp - dy
            x_start = self.max_disp - dx
            x_2_slice = x_2_padded[:, :, y_start: y_start + h, x_start: x_start + w]
            corr = torch.mean(x_1 * x_2_slice, dim=1, keepdim=True)
            correlations.append(corr)
        x_out = torch.cat(correlations, dim=1)
        return x_out



def split_images(x, patch_size):
    n, height, width, channels = x.shape
    fh, fw = patch_size

    grid_height = height // fh
    grid_width = width // fw

    x_rearranged = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=fh, fw=fw)
    return x_rearranged


def unsplit_images(x, grid_size, patch_size):
    gh, gw = grid_size
    fh, fw = patch_size
    x_rearranged = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=gh, gw=gw, fh=fh, fw=fw)
    return x_rearranged


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W = 2, 1, 10, 10

    model = CorrMLP(in_channels=C, enc_channels=8, dec_channels=16, use_checkpoint=False).to(device)
    model.eval()

    fixed_img = torch.randn(B, C, H, W, device=device)
    moving_img = torch.randn(B, C, H, W, device=device)

    num_params, storage_size = calculate_parameters(model)
    storage_size_mb = storage_size / (1024 ** 2)

    print(f'Total parameters: {num_params}')
    print(f'Model storage size: {storage_size_mb:.2f} MB')