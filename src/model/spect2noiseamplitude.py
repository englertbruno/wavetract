#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------

import torch
import torch.nn as nn


class Spect2NoiseAmplitude(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(Spect2NoiseAmplitude, self).__init__()

        channels = in_channels
        # self.project = nn.Sequential(
        #     DoubleConv(channels, channels, residual=True),
        #     DoubleConv(channels, channels, residual=True),
        #     DoubleConv(channels, channels, residual=True),
        # )
        #
        # upscale = []
        # channels = channels
        # for _ in range(scale_factor):
        #     new_channels = max(16, channels // 4)
        #     upscale += [Up(channels, new_channels)]
        #     channels = new_channels
        # self.upscale = nn.ModuleList(upscale)
        # self.norm = nn.BatchNorm1d(channels)

        out = torch.nn.Conv1d(channels, out_channels, kernel_size=7, stride=1, padding=3)

        torch.nn.init.normal_(out.weight, 0, std=0.1)
        torch.nn.init.constant_(out.bias, -1.0)

        self.out = out

    def forward(self, x):
        # x = self.project(x)
        #
        # for up in self.upscale:
        #     x = up(x)
        # x = self.norm(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x
