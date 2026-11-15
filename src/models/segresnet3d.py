"""
Pure PyTorch 3D SegResNet implementation bit-for-bit compatible with MONAI SegResNet weights.

Default configuration (init_filters=32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
matches ~18.80M parameters (18,795,969) without requiring MONAI at inference time.
"""

import torch
import torch.nn as nn


class _SegConvWrap(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _SegResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c)
        self.norm2 = nn.GroupNorm(8, c)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = _SegConvWrap(c, c, 3, 1, 1, False)
        self.conv2 = _SegConvWrap(c, c, 3, 1, 1, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return x + identity


class _SegUpSampleWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample_non_trainable = nn.Upsample(
            scale_factor=(2.0, 2.0, 2.0), mode="trilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample_non_trainable(x)


class PureSegResNet3D(nn.Module):
    """3D SegResNet architecture matching MONAI SegResNet state_dict keys without external dependencies."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_filters: int = 32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    ):
        super().__init__()
        self.act_mod = nn.ReLU(inplace=True)
        self.convInit = _SegConvWrap(in_channels, init_filters, 3, 1, 1, False)
        self.down_layers = nn.ModuleList()

        c = init_filters
        for i, n_blk in enumerate(blocks_down):
            layers = []
            if i == 0:
                layers.append(nn.Identity())
            else:
                layers.append(_SegConvWrap(c // 2, c, 3, 2, 1, False))
            for _ in range(n_blk):
                layers.append(_SegResBlock(c))
            self.down_layers.append(nn.Sequential(*layers))
            if i < len(blocks_down) - 1:
                c *= 2

        self.up_layers = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for _, n_blk in enumerate(blocks_up):
            half_c = c // 2
            self.up_samples.append(
                nn.Sequential(_SegConvWrap(c, half_c, 1, 1, 0, False), _SegUpSampleWrap())
            )
            self.up_layers.append(nn.Sequential(*[_SegResBlock(half_c) for _ in range(n_blk)]))
            c = half_c

        self.conv_final = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.ReLU(inplace=True),
            _SegConvWrap(c, out_channels, 1, 1, 0, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convInit(x)
        skips = []
        for down in self.down_layers:
            x = down(x)
            skips.append(x)
        skips.pop()
        for up_s, up_l in zip(self.up_samples, self.up_layers):
            x = up_s(x) + skips.pop()
            x = up_l(x)
        return self.conv_final(x)
