import torch
import torch.nn as nn
from monai.networks.blocks import Convolution

class UpSkipBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = Convolution(
            spatial_dims=3,
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            act="LEAKYRELU",
            norm="INSTANCE",
        )

    def forward(self, x, skip):
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class Decoder3D(nn.Module):
    def __init__(self, in_channels=768, out_channels=1, feature_size=48):
        super().__init__()
        self.bottleneck = nn.Conv3d(in_channels, feature_size * 16, kernel_size=1)

        self.up3 = UpSkipBlock3D(feature_size * 16, feature_size * 8, feature_size * 8)
        self.up2 = UpSkipBlock3D(feature_size * 8, feature_size * 4, feature_size * 4)
        self.up1 = UpSkipBlock3D(feature_size * 4, feature_size * 2, feature_size * 2)
        self.up0 = UpSkipBlock3D(feature_size * 2, feature_size, feature_size)

        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            Convolution(
                spatial_dims=3,
                in_channels=feature_size,
                out_channels=feature_size,
                kernel_size=3,
                strides=1,
                padding=1,
                act="LEAKYRELU",
                norm="INSTANCE",
            ),
        )

        self.out_conv = nn.Conv3d(feature_size, out_channels, kernel_size=1)

    def forward(self, x, skips):
        x = self.bottleneck(x)
        x = self.up3(x, skips[3])
        x = self.up2(x, skips[2])
        x = self.up1(x, skips[1])
        x = self.up0(x, skips[0])

        x = self.up_final(x)
        return self.out_conv(x)