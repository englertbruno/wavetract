#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
import torch
import torch.nn as nn


class Spect2Reflection(nn.Module):
    def __init__(self, in_channels, tract_length, scale_factor):
        super(Spect2Reflection, self).__init__()
        channels = in_channels
        self.max_diameter = 3.

        # self.project = nn.Sequential(
        #     DoubleConv(channels, channels, residual=True),
        #     DoubleConv(channels, channels, residual=True),
        #     DoubleConv(channels, channels, residual=True),
        # )
        # upscale = []
        # channels = channels
        # for _ in range(scale_factor):
        #     new_channels = max(16, channels // 4)
        #     upscale += [Up(channels, new_channels)]
        #     channels = new_channels
        # self.upscale = nn.ModuleList(upscale)
        # self.norm = nn.BatchNorm1d(channels)

        out_diameter = torch.nn.Conv1d(channels, tract_length, kernel_size=7, stride=1, padding=3)
        torch.nn.init.normal_(out_diameter.weight, 0, std=0.1)
        torch.nn.init.constant_(out_diameter.bias, 1)
        self.out_diameter = out_diameter

        out_end_reflection = torch.nn.Conv1d(channels, 2, kernel_size=7, stride=1, padding=3)
        torch.nn.init.normal_(out_end_reflection.weight, 0, std=0.1)
        torch.nn.init.constant_(out_end_reflection.bias, -3)
        self.out_end_reflection = out_end_reflection

        self.diameter = nn.Parameter(torch.ones((1, tract_length, 1)))

    def forward(self, x):
        # x = self.project(x)
        #
        # for up in self.upscale:
        #     x = up(x)
        # x = self.norm(x)

        out_diameter = self.out_diameter(x)
        out_end_reflection = self.out_end_reflection(x)
        out_end_reflection = torch.tanh(out_end_reflection)
        glottis_reflection, lip_reflection = out_end_reflection[:, 0:1, :], out_end_reflection[:, 1:2, :]

        out_diameter = (torch.sigmoid(out_diameter) * torch.sigmoid(self.diameter) * self.max_diameter) + 1e-4
        sa = out_diameter ** 2  # surface area
        tract_reflection = (sa[:, :-1, :] - sa[:, 1:, :]) / (sa[:, :-1, :] + sa[:, 1:, :] + 1e-6)

        reflections = torch.cat([glottis_reflection, tract_reflection, lip_reflection], dim=1)

        return reflections, out_diameter
