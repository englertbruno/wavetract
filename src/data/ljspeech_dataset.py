#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------

import os
import random
from os.path import isfile, join

import torch.utils.data

from src.data.file_util import load_wav_to_torch, load_mp3_to_torch

MAX_WAV_VALUE = 32768.0


class LJSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root):

        root = join(root, "wavs")
        self.files = [join(root, f) for f in os.listdir(root) if isfile(join(root, f))]

        files_compressed = [f.replace("wavs", "mp3s") for f in self.files]
        files_compressed = [f.replace("wav", "mp3") for f in files_compressed]
        self.files_compressed = files_compressed

        self.segment_length = 32768

    def __getitem__(self, index):
        while True:
            audio, _ = load_wav_to_torch(self.files[index])

            if audio.size(0) >= self.segment_length:
                audio_compressed, _ = load_mp3_to_torch(self.files_compressed[index])

                max_audio_start = audio.size(0) - self.segment_length
                audio_start = random.randint(0, max_audio_start)

                audio = audio[audio_start:audio_start + self.segment_length]
                audio_compressed = audio_compressed[audio_start:audio_start + self.segment_length]

                audio = audio  # / MAX_WAV_VALUE
                audio_compressed = audio_compressed  # / MAX_WAV_VALUE
                return audio.unsqueeze(0), audio_compressed.unsqueeze(0)

            index += 1

    def __len__(self):
        return len(self.files)

    @staticmethod
    def get_sample_rate():
        return 22050
