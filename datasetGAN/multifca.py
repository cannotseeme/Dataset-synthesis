import torch.nn as nn
import torch.nn.functional as F
from layer import MultiSpectralAttentionLayer
import numpy as np
import torch

class MultiFca(nn.Module):
    def __init__(self):
        super(MultiFca, self).__init__()
        self.ASPP = ASPP(2304, 2304)  # 2304 = 512 * 4 + 256 (pos)
        self.upsampler0 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.upsampler1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.upsampler2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upsampler3 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)
        self.upsampler4 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.upsampler5 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.upsampler6 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.row_embed = nn.Parameter(torch.rand(32, 256 // 2))
        self.col_embed = nn.Parameter(torch.rand(32, 256 // 2))

    def forward(self, x):
        out_0 = self.upsampler0(x[0])
        out_1 = self.upsampler1(x[1])
        out_2 = self.upsampler2(x[2])
        out_3 = x[3]
        # positional embedding
        H, W = out_3.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).unsqueeze(0).permute(0, 3, 1, 2)

        out_3 = torch.cat((out_0, out_1, out_2, out_3, pos), 1)
        # ASPP
        out_3 = self.ASPP(out_3)

        out_3 = self.upsampler3(out_3)
        out_4 = self.upsampler4(x[4])
        out_5 = self.upsampler5(x[5])
        out_6 = self.upsampler6(x[6])
        out_7 = x[7]
        out = torch.cat((out_3, out_4, out_5, out_6, out_7), 1)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, small=False):
        super(FPN, self).__init__()
        self.small = small
        self.project = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False),
        )
        self.path0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False),
        )
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False),
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=4, dilation=4, bias=False),
        )
        self.path3 = nn.Sequential(
            FcaBasicBlock(in_channel, out_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False),
        )

    def forward(self, x):
        x_0 = self.path0(x)
        x_1 = self.path1(x)
        x_2 = self.path2(x)
        x_3 = self.path3(x)
        x_c = self.conv1(x)
        x = x_0 + x_1 + x_2 + x_3 + x_c
        return self.project(x)

class FcaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(FcaBasicBlock, self).__init__()
        c2wh = dict([(992, [512, 512]), (2528, [512, 512]), (512, [64, 64]), (256, [128, 128]), (128, [256, 256]), (64, [512, 512]),
                     (32, [512, 512]),
                     (2048, [32, 32]), (2304, [32, 32]), (1280, [32, 32]), (2784, [512, 512]), (3264, [512, 512]),])
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(planes, c2wh[planes][0], c2wh[planes][1],  reduction=reduction, freq_sel_method='top16')
        self.downsample = downsample
        self.stride = stride
        self.conv_res = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        residual = self.conv_res(residual)
        out += residual
        out = self.relu(out)

        return out