#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------

import wave


def save_torch2wav(x, sample_rate, out_path):
    audio = x.cpu().numpy()
    audio = (audio * (2 ** 15 - 1)).astype("<h")
    with wave.open(out_path, "w") as f:
        # Mono Channel.
        f.setnchannels(1)
        # 2 bytes per sample.
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio.tobytes())
