#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
from torch import nn


class LayerNorm1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, mid_channels, kernel_size=5, padding=2, bias=True),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=5, padding=2, bias=True),
        )

    def forward(self, x):
        if self.residual:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, residual=(in_channels == out_channels)),
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Down(nn.Module):
    """Downscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Upsample(scale_factor=0.5, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, residual=(in_channels == out_channels)),
        )

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x
