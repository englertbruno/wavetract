#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------

import matplotlib

matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import soundfile as sf
import tempfile
import subprocess

from src.data.ears_dataset import EARSDataset
from src.model.neural_tract import NeuralTract


def main():
    # Training params
    seed = 123456
    use_amp = True
    device = 'cuda'

    # Data params
    cutoff = 6000.
    segment_length = 32768 * 4  # increase length compared to training
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

    test_dataset = (
        EARSDataset(
            os.path.abspath("out/ears"),
            segment_length=segment_length,
            cutoff=cutoff,
            n_mels=n_mel_channels,
            mode="test"
        )
    )

    model = NeuralTract(
        spect_net_n_blocks,
        tract_n, extra_simulation_steps, simulation_double,
        n_mel_channels, sample_rate, cutoff
    )
    model.load_state_dict(torch.load(os.path.abspath("out/20250326_123419_4d405184_finished/ckpt/neuraltract_40000.pth")))
    model = model.to(device)
    model.eval()

    audio, audio_compressed, spect = test_dataset[np.random.randint(0, len(test_dataset))]
    audio, audio_compressed, spect = audio.to(device), audio_compressed.to(device), spect.to(device)
    audio, audio_compressed, spect = audio.unsqueeze(0), audio_compressed.unsqueeze(0), spect.unsqueeze(0)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            audio_uncompressed, _, diameters = model(audio_compressed, spect)
    audio_uncompressed = audio_uncompressed.squeeze().cpu().numpy()
    audio_uncompressed = (audio_uncompressed * (2 ** 15 - 1)).astype("<h")
    n_samples = audio_uncompressed.shape[-1]
    time_length = n_samples / sample_rate

    # Parameters
    fps = 30
    n_frames = int(fps * time_length)
    segment_length = 1.0  # arbitrary units
    x = np.arange(tract_n) * segment_length

    diameters = diameters.cpu()
    diameters = torch.nn.functional.interpolate(diameters, size=n_frames, mode="linear", align_corners=True)
    diameters = diameters.permute(0, 2, 1).numpy()[0]
    print(diameters.shape)
    print(diameters.min(), diameters.max())

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    line_top, = ax.plot([], [], color='blue')
    line_bottom, = ax.plot([], [], color='blue')
    ax.set_xlim(0, tract_n * segment_length)
    ax.set_ylim(-np.max(diameters) * 1.2, np.max(diameters) * 1.2)
    ax.set_xlabel("Tract Length")
    ax.set_ylabel("Diameter")
    ax.set_title("Vocal Tract Shape Over Time")

    def init():
        line_top.set_data([], [])
        line_bottom.set_data([], [])
        return line_top, line_bottom

    def animate(i):
        y_top = diameters[i] / 2
        y_bottom = -diameters[i] / 2
        line_top.set_data(x, y_top)
        line_bottom.set_data(x, y_bottom)
        return line_top, line_bottom

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=1000 / fps, blit=True
    )

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "video.mp4")
    audio_path = os.path.join(temp_dir, "audio.wav")
    output_path = "../englertbruno.github.io/posts/003_neural_tract/audio/vocal_tract_with_audio.mp4"

    # Save the animation
    ani.save(video_path, fps=fps, dpi=200)
    plt.close()
    print("Animation complete!")

    sf.write(audio_path, audio_uncompressed, sample_rate)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]

    subprocess.run(cmd, check=True)
    print(f"✅ Video with audio saved as: {output_path}")


if __name__ == "__main__":
    main()
