#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
import time

import matplotlib

matplotlib.use('Agg')

import os
from os.path import join
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import git

from src.data.ears_dataset import EARSDataset
from src.util.get_lr import get_lr
from src.util.mkdir import mkdir
from src.util.viz import viz
from src.util.spectrogram import Spectrogram
from src.model.neural_tract import NeuralTract
from src.util.save_torch2wav import save_torch2wav


def main():
    # Prepare project space
    git_sha = git.Repo(search_parent_directories=True).head.object.hexsha
    git_sha = git_sha[:8]
    timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    spect_path = f"out/{timestamp}_{git_sha}/spectrogram"
    voice_path = f"out/{timestamp}_{git_sha}/voice"
    ckpt_path = f"out/{timestamp}_{git_sha}/ckpt"
    mkdir(spect_path)
    mkdir(voice_path)
    mkdir(ckpt_path)

    # Training params
    seed = 1234
    base_lr = 5e-5
    final_lr = 1e-7
    warmup_iters = 100
    batch_size = 2
    max_iters = 40000
    accumulation_iters = 2
    use_amp = True
    device = 'cuda'

    # Data params
    cutoff = 6000.
    segment_length = 32768
    sample_rate = EARSDataset.get_sample_rate()

    # Tract params
    tract_n = 40
    extra_simulation_steps = 60
    simulation_double = True

    # Model params
    n_mel_channels = 80
    spect_net_n_blocks = 8

    # Fix seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # print("train_dataset len:", len(train_dataset))
    spectrogram = Spectrogram(n_fft=128, hop=64)
    train_dataset = DataLoader(
        EARSDataset(
            os.path.abspath("out/ears"),
            segment_length=segment_length,
            cutoff=cutoff,
            n_mels=n_mel_channels,
            mode="train"
        ),
        batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True, drop_last=True
    )

    def get_batch():
        while True:
            for batch in train_dataset:
                audio, audio_compressed, spect = batch
                audio = audio.pin_memory().to(device, non_blocking=True)
                audio_compressed = audio_compressed.pin_memory().to(device, non_blocking=True)
                spect = spect.pin_memory().to(device, non_blocking=True)
                yield (audio, audio_compressed, spect)

    test_dataset = (
        EARSDataset(
            os.path.abspath("out/ears"),
            segment_length=segment_length,
            cutoff=cutoff,
            n_mels=n_mel_channels,
            mode="val"
        )
    )

    model = NeuralTract(
        spect_net_n_blocks,
        tract_n, extra_simulation_steps, simulation_double,
        n_mel_channels, sample_rate, cutoff
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.spect_net.parameters())
    print("spect_net number of parameters: %.2fM" % (n_params / 1e6,))
    n_params = sum(p.numel() for p in model.spect2reflection.parameters())
    print("spect2admittance number of parameters: %.2fM" % (n_params / 1e6,))
    n_params = sum(p.numel() for p in model.spect2noise_amplitude.parameters())
    print("spect2noise_amplitude number of parameters: %.2fM" % (n_params / 1e6,))
    n_params = sum(p.numel() for p in model.upscale_spect.parameters())
    print("upscale_spect number of parameters: %.2fM" % (n_params / 1e6,))

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    get_batch_iterator = iter(get_batch())
    train_iter = 0
    start_time = time.time()
    model.train()
    while train_iter <= max_iters:
        loss_accumulation = 0.
        opt.zero_grad()

        for _ in range(accumulation_iters):
            # Load data
            audio, audio_compressed, spect = next(get_batch_iterator)

            # Run model
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                audio_uncompressed, noise, tract_diameters = model(audio_compressed, spect)

            # Get spect
            spec_pred = torch.log(torch.abs(spectrogram(audio_uncompressed.squeeze(1)) + 1e-7))
            spec_label = torch.log(torch.abs(spectrogram(audio.squeeze(1)) + 1e-7))
            spec_comp = torch.log(torch.abs(spectrogram(audio_compressed.squeeze(1)) + 1e-7))

            loss_diff = F.mse_loss(spec_pred, spec_label) + F.l1_loss(audio_uncompressed, audio) * 0.1
            loss_diff_ref = F.mse_loss(spec_comp, spec_label) + F.l1_loss(audio_compressed, audio) * 0.1
            variational_loss = torch.mean(torch.abs(tract_diameters[:, 1:, :] - tract_diameters[:, :-1, :])) * 0.01
            loss = loss_diff + variational_loss

            # Backward pass with grad accumulation
            scaler.scale(loss / accumulation_iters).backward()

        # Update LR
        lr = get_lr(train_iter, warmup_iters, max_iters, base_lr, final_lr)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        # Step optimizer
        scaler.step(opt)
        scaler.update()

        # Visualization, eval
        if (train_iter % 5) == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_iter = elapsed_time / (train_iter + 1)
            remaining_iters = max_iters - train_iter - 1
            estimated_remaining_time = remaining_iters * avg_time_per_iter
            remaining_td = timedelta(seconds=estimated_remaining_time)

            print(
                "{:09d}:\t loss: {:.3f}, loss ref: {:.3f}, var loss: {:.3f} lr: {:.2e}, Estimated time remaining: {}".format(
                    train_iter,
                    loss_diff.item(), loss_diff_ref.item(), variational_loss.item(),
                    lr, str(remaining_td).split('.')[0]
                )
            )
        if (train_iter % 50) == 0:
            viz(
                audio_compressed, noise, audio_uncompressed, audio,
                train_iter, sample_rate, out_path=join(spect_path, f"{train_iter}.png")
            )

        #
        if max_iters < train_iter or (train_iter % 1000) == 0:
            torch.save(model.state_dict(), join(ckpt_path, f"neuraltract_{train_iter}.pth"))

        #
        if (train_iter % 200) == 0:
            model.eval()

            audio, audio_compressed, spect = test_dataset[np.random.randint(0, len(test_dataset))]
            audio, audio_compressed, spect = audio.to(device), audio_compressed.to(device), spect.to(device)
            audio, audio_compressed, spect = audio.unsqueeze(0), audio_compressed.unsqueeze(0), spect.unsqueeze(0)

            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    audio_uncompressed, _, _ = model(audio_compressed, spect)

            save_torch2wav(audio_uncompressed, sample_rate, join(voice_path, f"{train_iter}_uncompressed.wav"))
            save_torch2wav(audio_compressed, sample_rate, join(voice_path, f"{train_iter}_compressed.wav"))
            save_torch2wav(audio, sample_rate, join(voice_path, f"{train_iter}_orig.wav"))

            model.train()

        train_iter += 1
    os.rename(f"out/{timestamp}_{git_sha}", f"out/{timestamp}_{git_sha}_finished")


if __name__ == "__main__":
    main()
