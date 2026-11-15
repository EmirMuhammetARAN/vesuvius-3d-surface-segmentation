"""
3D Residual Encoder U-Net (ResEncUNet 3D) matching the nnU-Net architecture.
Designed for memory-efficient training on 12GB VRAM with FP16/AMP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act2(x + res)
        return x


class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False)
        self.norm_up = nn.InstanceNorm3d(in_channels // 2, affine=True)
        self.act_up = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        total_in = (in_channels // 2) + skip_channels
        self.conv = ResidualBlock3D(total_in, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act_up(self.norm_up(self.upsample(x)))
        # Pad if shapes mismatch slightly due to odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResEncUNet3D(nn.Module):
    """
    3D U-Net with Residual Encoder blocks.
    Default channels: [32, 64, 128, 256, 320]
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: list = [32, 64, 128, 256, 320]
    ):
        super().__init__()
        c0, c1, c2, c3, c4 = channels

        # Encoder (Down)
        self.enc0 = ResidualBlock3D(in_channels, c0)
        self.down0 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc1 = ResidualBlock3D(c0, c1)
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ResidualBlock3D(c1, c2)
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ResidualBlock3D(c2, c3)
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(c3, c4)

        # Decoder (Up)
        self.dec3 = DecoderBlock3D(c4, c3, c3)
        self.dec2 = DecoderBlock3D(c3, c2, c2)
        self.dec1 = DecoderBlock3D(c2, c1, c1)
        self.dec0 = DecoderBlock3D(c1, c0, c0)

        # Final prediction head (1x1x1 Conv)
        self.out_conv = nn.Conv3d(c0, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)              # (B, c0, D, H, W)
        d0 = self.down0(e0)             # (B, c0, D/2, H/2, W/2)

        e1 = self.enc1(d0)             # (B, c1, D/2, H/2, W/2)
        d1 = self.down1(e1)             # (B, c1, D/4, H/4, W/4)

        e2 = self.enc2(d1)             # (B, c2, D/4, H/4, W/4)
        d2 = self.down2(e2)             # (B, c2, D/8, H/8, W/8)

        e3 = self.enc3(d2)             # (B, c3, D/8, H/8, W/8)
        d3 = self.down3(e3)             # (B, c3, D/16, H/16, W/16)

        # Bottleneck
        b = self.bottleneck(d3)         # (B, c4, D/16, H/16, W/16)

        # Decoder
        u3 = self.dec3(b, e3)           # (B, c3, D/8, H/8, W/8)
        u2 = self.dec2(u3, e2)          # (B, c2, D/4, H/4, W/4)
        u1 = self.dec1(u2, e1)          # (B, c1, D/2, H/2, W/2)
        u0 = self.dec0(u1, e0)          # (B, c0, D, H, W)

        logits = self.out_conv(u0)       # (B, out_channels, D, H, W)
        return logits
