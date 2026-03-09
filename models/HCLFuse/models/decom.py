import torch
import torch.nn as nn
import warnings
import os
import math
import torch.nn.functional as F
from einops import rearrange

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()

        sequence = []

        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        ]

        self.model = nn.Sequential(*sequence)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        out = self.model(x) + self.conv(x)

        return out


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out


class channel_down(nn.Module):
    def __init__(self, channels):
        super(channel_down, self).__init__()

        self.conv0 = nn.Conv2d(channels * 4, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels, 2, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = torch.sigmoid(self.conv2(self.relu(self.conv1(self.relu(self.conv0(x))))))

        return out


class channel_up(nn.Module):
    def __init__(self, channels):
        super(channel_up, self).__init__()

        self.conv0 = nn.Conv2d(2, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(self.relu(self.conv0(x)))))

        return out

class Mask(nn.Module):
    def __init__(self, size):
        super(Mask, self).__init__()
        self.mask = nn.Parameter(torch.randn(size, requires_grad=True))

    def forward(self, x):
        return torch.sigmoid(self.mask) * x

class feature_pyramid(nn.Module):
    def __init__(self,B, channels):
        super(feature_pyramid, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.Conv2d(channels, channels, kernel_size=(5, 5), stride=(1, 1), padding=2)
        )

        self.block0 = Res_block(channels, channels)
        self.down0 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.mask0 = Mask(size=(1, channels, 1, 1))

        self.block1 = Res_block(channels, channels * 2)
        self.down1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.mask1 = Mask(size=(1, channels * 2, 1, 1))

        self.block2 = Res_block(channels * 2, channels * 4)
        self.down2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.mask2 = Mask(size=(1, channels * 4, 1, 1))

        self.relu = nn.LeakyReLU()

        self.feature = {}

    def forward(self, x):
        x1 = self.convs(x)
        x1_masked = self.mask0(self.block0(x1))
        level0 = self.down0(x1_masked)

        x2 = self.block1(level0)
        x2_masked = self.mask1(x2)
        level1 = self.down1(x2_masked)

        x3 = self.block2(level1)
        x3_masked = self.mask2(x3)
        level2 = self.down2(x3_masked)
        
        self.feature['down1']=level0
        self.feature['down2']=level1
        self.feature['down3']=level2

        return level2

    def get_features(self):
        return self.feature

class ReconNet(nn.Module):
    def __init__(self,B, channels):
        super(ReconNet, self).__init__()

        self.pyramid = feature_pyramid(B,channels)

        self.channel_down = channel_down(channels)
        self.channel_up = channel_up(channels)
        self.block_up0 = Res_block(channels * 4, channels * 4)
        self.block_up1 = Res_block(channels * 4, channels * 4)
        self.up_sampling0 = upsampling(channels * 4, channels * 2)
        self.block_up2 = Res_block(channels * 2, channels * 2)
        self.block_up3 = Res_block(channels * 2, channels * 2)
        self.up_sampling1 = upsampling(channels * 2, channels)
        self.block_up4 = Res_block(channels, channels)
        self.block_up5 = Res_block(channels, channels)
        self.up_sampling2 = upsampling(channels, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.relu = nn.LeakyReLU()
    def forward(self, x, pred_fea=None,features_down=None):
        if pred_fea is None:
            z = self.pyramid(x[:, :2, ...])
            features = self.pyramid.get_features()
            z = self.channel_down(z)
            mu = torch.mean(z, dim=[2, 3], keepdim=True)
            logvar = torch.log(torch.var(z, dim=[2, 3], keepdim=True) + 1e-6)

            return z,mu,logvar, features
        else:
            pred_fea = self.channel_up(pred_fea)

            pred_fea_up2 = self.up_sampling0(
                self.block_up1(self.block_up0(pred_fea) + features_down['down3']))
            pred_fea_up4 = self.up_sampling1(
                self.block_up3(self.block_up2(pred_fea_up2) + features_down['down2']))
            pred_fea_up8 = self.up_sampling2(
                self.block_up5(self.block_up4(pred_fea_up4) + features_down['down1']))

            pred_img = self.conv3(self.relu(self.conv2(pred_fea_up8)))

            return pred_img



class Encoder(nn.Module):
    def __init__(self,config, channels=64):
        super(Encoder, self).__init__()

        self.ReconNet = ReconNet(config.training.batch_size,channels)

    def forward(self, images, pred_fea=None,features_down=None):

        output = {}
        if pred_fea is None:
            z,mu,logvar, features = self.ReconNet(images, pred_fea=None,features_down=None)

            output["z"] = z
            output["mu"] = mu
            output["logvar"] = logvar
            output["features"] = features
        else:
            pred_img = self.ReconNet(images, pred_fea=pred_fea,features_down=features_down)
            output["pred_img"] = pred_img
        return output
