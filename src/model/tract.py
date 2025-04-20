#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
import torch
from torch import nn


class Tract(nn.Module):
    def __init__(self, tract_n, extra_simulation_steps, simulation_double=False):
        super(Tract, self).__init__()
        self.tract_n = tract_n
        self.simulation_double = simulation_double
        self.simulation_steps = extra_simulation_steps
        self.fade = nn.Parameter(torch.ones((1, 1, self.tract_n)) * 10)

        self.pad_right = torch.nn.ConstantPad1d((1, 0), 0)
        self.pad_left = torch.nn.ConstantPad1d((0, 1), 0)
        self.pad_admittance = torch.nn.ConstantPad1d((0, self.tract_n + self.simulation_steps), 0)

    def forward(self, in_audio, reflection):
        # torch.Size([2, 1, 32892])
        if self.simulation_double:
            in_audio = torch.nn.functional.interpolate(in_audio, scale_factor=2., mode='nearest')
            reflection = torch.nn.functional.interpolate(reflection, scale_factor=2., mode='linear')
        batch_size, _, audio_length = in_audio.size()
        reflection = self.pad_admittance(reflection)

        assert (reflection.size(2) >= audio_length + self.tract_n + self.simulation_steps), (
            "{}, {} + {} + {}".format(
                reflection.size(2), audio_length, self.tract_n, self.simulation_steps,
            ))

        tract_r = torch.zeros((batch_size, audio_length, self.tract_n)).to(reflection)
        tract_l = torch.zeros((batch_size, audio_length, self.tract_n)).to(reflection)

        tract_r[:, :, 0:1] = in_audio.permute(0, 2, 1)
        reflection = reflection.permute(0, 2, 1)

        def step(i, tract_r, tract_l):
            shifted_r = self.pad_right(tract_r[..., :-1])
            new_tract_r = shifted_r - (shifted_r + tract_l) * reflection[:, i:i + audio_length, :self.tract_n]

            shifted_l = self.pad_left(tract_l[..., 1:])
            new_tract_l = shifted_l + (shifted_l + tract_r) * reflection[:, i:i + audio_length, 1:]
            new_tract_r = new_tract_r * torch.sigmoid(self.fade)
            new_tract_l = new_tract_l * torch.sigmoid(self.fade)
            return new_tract_r, new_tract_l

        for i in range(self.tract_n):
            tract_r, tract_l = step(i, tract_r, tract_l)
        output_audio = tract_r[..., -1]
        for r in range(self.simulation_steps):
            i = r + self.tract_n
            tract_r, tract_l = step(i, tract_r, tract_l)
            output_audio[:, r + 1:] += tract_r[:, :audio_length - r - 1, -1]

        output_audio = output_audio.unsqueeze(1)

        if self.simulation_double:
            output_audio = output_audio[..., ::2] + output_audio[..., 1::2]
        return output_audio
