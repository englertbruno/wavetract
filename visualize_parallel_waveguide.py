#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    # ==== Configurable Parameters ====
    tube_length = 40
    n_frames = 120

    n_parallel = 4  # Number of parallel simulations (tubes)
    n_ar_dots = 4  # Number of autoregressive dots
    dot_interval_ar = 5  # Emit a new AR dot every N frames
    dot_speed = 0.5  # Dot speed (positions per frame)

    dot_decay = 0.9
    tube_amplitude = 0.4

    # ==== Setup Figure ====
    fig, axs = plt.subplots(n_parallel + 1, 1, figsize=(10, 2 + 2 * n_parallel), sharex=True)
    fig.suptitle("Digital Waveguide: Autoregressive vs Parallelized", fontsize=14)

    # ==== Autoregressive Tube ====
    ax_ar = axs[0]
    ax_ar.set_xlim(0, tube_length)
    ax_ar.set_ylim(-1.0, 1.0)
    ax_ar.set_yticks([])
    ax_ar.set_xticks([])
    ax_ar.set_title("Autoregressive (Single Tube, Step-by-Step)")

    tube_top_ar, = ax_ar.plot([], [], color='black')
    tube_bot_ar, = ax_ar.plot([], [], color='black')
    dots_ar, = ax_ar.plot([], [], 'o', color='blue')

    # Track moving pressure pulses
    ar_dots = []

    # ==== Parallel Tubes ====
    tube_lines_top = []
    tube_lines_bot = []
    dots_parallel = []
    for i in range(n_parallel):
        ax = axs[i + 1]
        ax.set_xlim(0, tube_length)
        ax.set_ylim(-1.0, 1.0)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"Parallel Tube {i + 1}")

        top_line, = ax.plot([], [], color='black')
        bot_line, = ax.plot([], [], color='black')
        dot, = ax.plot([], [], 'o', color='green')

        tube_lines_top.append(top_line)
        tube_lines_bot.append(bot_line)
        dots_parallel.append(dot)

    # ==== Tube Shape Generator ====
    def get_rich_tube_diameter(t, length, phase_shift=0.0):
        x = np.linspace(0, tube_length - 1, length)
        base = 0.3
        component1 = 0.15 * np.sin(2 * np.pi * (x / tube_length + t / 40.0 + phase_shift))
        component2 = 0.05 * np.sin(2 * np.pi * (3 * x / tube_length + t / 20.0 + phase_shift))
        component3 = 0.03 * np.sin(2 * np.pi * (6 * x / tube_length + t / 10.0 + phase_shift))
        component4 = 0.02 * np.sin(2 * np.pi * (12 * x / tube_length + t / 5.0 + phase_shift))
        diameter = base + component1 + component2 + component3 + component4
        return x, diameter

    # ==== Init Function ====
    def init():
        tube_top_ar.set_data([], [])
        tube_bot_ar.set_data([], [])
        dots_ar.set_data([], [])
        for i in range(n_parallel):
            tube_lines_top[i].set_data([], [])
            tube_lines_bot[i].set_data([], [])
            dots_parallel[i].set_data([], [])
        return [tube_top_ar, tube_bot_ar, dots_ar] + tube_lines_top + tube_lines_bot + dots_parallel

    # ==== Animate ====
    def animate(frame):
        # === Autoregressive ===
        x_ar, diameter = get_rich_tube_diameter(frame, tube_length)
        tube_top_ar.set_data(x_ar, diameter)
        tube_bot_ar.set_data(x_ar, -diameter)

        # Emit up to n_ar_dots total, spaced by dot_interval_ar
        if len(ar_dots) < n_ar_dots and frame % dot_interval_ar == 0:
            ar_dots.append({'pos': 0, 'amp': 1.0})

        # Update AR dots (remove those out of bounds)
        active_x = []
        active_y = []
        for dot in ar_dots:
            if dot['pos'] < tube_length:
                dot['pos'] += dot_speed
                dot['amp'] *= dot_decay
                if dot['pos'] < tube_length:
                    active_x.append(dot['pos'])
                    active_y.append(0.0)
        dots_ar.set_data(active_x, active_y)

        # === Parallel Tubes ===
        for i in range(n_parallel):
            phase = i * 0.4
            x_p, diameter_p = get_rich_tube_diameter(frame, tube_length, phase)
            tube_lines_top[i].set_data(x_p, diameter_p)
            tube_lines_bot[i].set_data(x_p, -diameter_p)

            # Compute dot lifespan based on speed
            dot_lifespan = tube_length // dot_speed + 1
            dot_start = 0  # i * 5
            if dot_start <= frame < dot_start + dot_lifespan:
                x_dot = (frame - dot_start) * dot_speed
                if x_dot < tube_length:
                    dots_parallel[i].set_data([x_dot], [0.0])
                else:
                    dots_parallel[i].set_data([], [])
            else:
                dots_parallel[i].set_data([], [])

        return [tube_top_ar, tube_bot_ar, dots_ar] + tube_lines_top + tube_lines_bot + dots_parallel

    # ==== Animate and Save ====
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=n_frames, interval=80, blit=True)

    video_path = os.path.join("./", "viz_parallel_waveguide.mp4")
    # Save the animation
    ani.save(video_path, fps=30, dpi=200)
    plt.close()
    print("Animation complete!")


if __name__ == "__main__":
    main()
