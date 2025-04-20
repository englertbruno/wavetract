#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
import torch
from torch import nn

from src.fir_filters.high_pass_filter import high_pass_filter
from src.fir_filters.low_pass_filter import low_pass_filter
from src.model.layers import Up
from src.model.spect2noiseamplitude import Spect2NoiseAmplitude
from src.model.spect2reflection import Spect2Reflection
from src.model.spect_net_conv import SpectNetConv
from src.model.tract import Tract


class NeuralTract(torch.nn.Module):
    def __init__(self, spect_net_n_blocks, tract_n, extra_simulation_steps, simulation_double, n_mels, sample_rate,
                 cutoff):
        super(NeuralTract, self).__init__()

        self.sample_rate = sample_rate
        self.tract_n = tract_n
        self.extra_simulation_steps = extra_simulation_steps
        self.embed_dim = 384
        # self.spectrogram = Spectrogram(n_fft=n_fft, hop=64, normalize=False, pad=n_fft // 4, center=False)

        self.conv_spect_preprocess = nn.Sequential(
            nn.Conv1d(n_mels, self.embed_dim, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(self.embed_dim),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=5, stride=1, padding=2),
            Up(self.embed_dim, self.embed_dim),
            Up(self.embed_dim, self.embed_dim),
        )

        self.spect_net = SpectNetConv(self.embed_dim, spect_net_n_blocks)
        self.upscale_spect = nn.Sequential(
            #
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.embed_dim),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1),
            #
            # Up(self.embed_dim, self.embed_dim),
            # Up(self.embed_dim, self.embed_dim // 2),
            Up(self.embed_dim, self.embed_dim // 4),
            Up(self.embed_dim // 4, self.embed_dim // 8),
            nn.BatchNorm1d(self.embed_dim // 8),
            nn.Conv1d(self.embed_dim // 8, self.embed_dim // 8, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=16, mode="linear"),
            nn.ReLU(),
        )

        self.spect2noise_amplitude = Spect2NoiseAmplitude(self.embed_dim // 8, 1, scale_factor=6)
        self.spect2reflection = Spect2Reflection(self.embed_dim // 8, self.tract_n, scale_factor=6)

        self.tract = Tract(
            tract_n=self.tract_n,
            extra_simulation_steps=extra_simulation_steps,
            simulation_double=simulation_double
        )

        out_conv_kernel_size = 101
        self.conv_out = nn.Sequential(
            nn.Conv1d(1, 1, stride=1,
                      kernel_size=out_conv_kernel_size,
                      padding=(out_conv_kernel_size - 1) // 2),
        )

        self.alpha_mix = nn.Parameter(torch.ones((1,)))
        self.alpha_out = nn.Parameter(torch.ones((1,)))

        self.filter_size = 16385
        high_pass = torch.from_numpy(high_pass_filter(f_c=cutoff - 10,
                                                      f_samp=float(self.sample_rate),
                                                      N=self.filter_size)).float()
        high_pass = high_pass.view(1, 1, self.filter_size).contiguous()
        self.register_buffer("high_pass", high_pass)
        low_pass = torch.from_numpy(low_pass_filter(f_c=cutoff + 10,
                                                    f_samp=float(self.sample_rate),
                                                    N=self.filter_size)).float()
        low_pass = low_pass.view(1, 1, self.filter_size).contiguous()
        self.register_buffer("low_pass", low_pass)

    def forward(self, in_audio_compressed, spect):
        b, c, n = in_audio_compressed.shape

        # Create white noise
        noise = torch.randn_like(in_audio_compressed) * 0.1

        # High and low pass filtering
        # in_audio_compressed = nn.functional.conv1d(in_audio, self.low_pass, bias=None, stride=1, padding="same")
        noise = nn.functional.conv1d(noise, self.high_pass, bias=None, stride=1, padding="same")

        # Process spectrogram
        spect = spect / 10.
        spect = self.conv_spect_preprocess(spect)

        # Run models
        alpha_mix = torch.sigmoid(self.alpha_mix)
        spect = self.upscale_spect(self.spect_net(spect))
        noise_amplitude = self.spect2noise_amplitude(spect)[:, :, :n]
        reflection, tract_diameters = self.spect2reflection(spect)

        # Run vocal tract simulation
        if False:
            noise = noise * noise_amplitude
            x = noise * (1. - alpha_mix) + in_audio_compressed * alpha_mix
            x = self.tract(x, admittance)
            x = x * self.alpha_out
            # x = self.conv_out(x)
        else:
            noise = noise * noise_amplitude
            x = noise
            x = self.tract(x, reflection)

            x = x * (1. - alpha_mix) + in_audio_compressed * alpha_mix
            x = x * self.alpha_out
            # x = self.conv_out(x)

        return x, noise, tract_diameters
