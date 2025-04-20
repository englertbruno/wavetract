#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
from scipy.signal import spectrogram
from matplotlib.patches import Rectangle


def main():
    video_fps = 30
    audio_path = 'out/ears/p001/emo_cuteness_sentences.wav'  # Must match the duration of your video
    output_filename = 'viz_spectrogram_vowels.mp4'

    # --- Load audio ---
    sr, audio = wavfile.read(audio_path)

    # Normalize if needed
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # --- Generate spectrogram ---
    f, t_spec, Sxx = spectrogram(audio, sr, nperseg=1024, noverlap=512)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    # --- Plot once: render as image ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_yticks(np.arange(0, sr / 2 + 1, 3000))
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    ax.set_ylim(0, sr / 2)
    ax.set_xlim(0, t_spec[-1])
    spec_img = ax.imshow(
        Sxx_dB,
        aspect='auto',
        extent=[t_spec[0], t_spec[-1], f[0], f[-1]],
        origin='lower',
        # cmap='magma'
    )

    # --- Add static lines and annotations ---
    # Horizontal line at 6 kHz
    ax.axhline(y=6000, color='white', linestyle='--', linewidth=1.5)

    # Word overlays
    words = [
        "Look at that", r"$\bf{c}$", "ute", "little", "kitty", "cat",
        "Oh my", "goodness",
        "she is",
        r"$\bf{sss}$", "o", "cute",
        r"Tha$\bf{t's}$ the",
        "c", "u", r"te$\bf{st}$",
        "thing I've ever seen"
    ]
    timestamps = [
        0.75, 1.33, 1.5, 1.9, 2.2, 2.5,
        3.5, 4.1,
        4.8,
        5.1, 5.3, 5.6,
        6.8,
        7.3, 7.4, 7.7,
        8.1
    ]

    for word, ts in zip(words, timestamps):
        ax.text(ts, sr / 2 - 1200, word, color='white', fontsize=6,
                ha='left', va='bottom', backgroundcolor='none')

    # --- Add red line ---
    red_line = ax.axvline(x=0, color='red', linewidth=2)

    # --- Add current times ---
    time_text_y_pos = 0  # s/2
    time_text = ax.text(0, time_text_y_pos, '0.00s', color='red', fontsize=8,
                        ha='center', va='bottom', backgroundcolor='white')

    # --- Red shaded rectangles for consonant sounds ---
    consonant_segments = [
        ("s", 1.33, 1.5),
        ("t", 5.0, 5.35),
        ("k", 6.9, 7.03),
        ("sh", 7.8, 7.95),
    ]

    for label, start, end in consonant_segments:
        rect = Rectangle((start, 6500), end - start, 15000, color='red', alpha=0.1)
        ax.add_patch(rect)

    # Needed for animation
    duration = len(audio) / sr
    num_frames = int(duration * video_fps)

    # --- Animation function ---
    def update(frame_idx):
        current_time = frame_idx / video_fps
        red_line.set_xdata([current_time, current_time])
        time_text.set_position((current_time, time_text_y_pos))
        time_text.set_text(f"{current_time:.2f}s")
        return red_line,

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, blit=True
    )
    plt.tight_layout()

    # --- Save with audio ---
    tmp_video = 'temp_video.mp4'
    print("Start animating...")
    ani.save(tmp_video, fps=video_fps, dpi=200)

    # --- Combine video + audio using ffmpeg ---
    os.system(
        f'ffmpeg -y -i {tmp_video} -i {audio_path} -c:v libx264 -c:a aac -strict experimental -shortest {output_filename}')
    os.remove(tmp_video)
    print("Complete!")


if __name__ == "__main__":
    main()
