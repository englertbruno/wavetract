#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------#


import random
from pathlib import Path

import torch.utils.data
import torchaudio
from scipy import signal

from src.data.file_util import load_wav_to_torch

# https://sp-uhh.github.io/ears_dataset/
MAX_WAV_VALUE = 32768.0


class EARSDataset(torch.utils.data.Dataset):
    def __init__(self, root, segment_length, cutoff, n_mels, mode):
        self.files = sorted([path.resolve() for path in Path(root).rglob('*.wav')])
        self.segment_length = segment_length
        self.cutoff = cutoff
        self.mode = mode
        self.train_ration = 0.7

        len_train = int(self.train_ration * len(self.files))
        if mode == "train":
            self.files = self.files[:len_train]
        else:
            self.files = self.files[len_train:]

        n_fft = 1024
        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=EARSDataset.get_sample_rate(),
            hop_length=256,
            n_fft=n_fft,
            win_length=n_fft,
            n_mels=n_mels,
            f_min=0.0,
            f_max=EARSDataset.get_sample_rate() // 2,  # cutoff,
            center=False,
            pad=512 - 64
        )

    def __getitem__(self, index):
        while True:
            audio, _ = load_wav_to_torch(self.files[index], sample_rate=EARSDataset.get_sample_rate())

            if audio.size(0) >= self.segment_length:
                max_audio_start = audio.size(0) - self.segment_length
                audio_start = random.randint(0, max_audio_start)

                if self.mode == "train":
                    audio = audio[audio_start:audio_start + self.segment_length]
                elif self.mode == "val":
                    limit = audio.size(0) - (audio.size(0) % self.segment_length)
                    limit = min(limit, self.segment_length * 8)  # without this, it can cause out of gpu memory
                    audio = audio[:limit]
                elif self.mode == "test":
                    limit = audio.size(0) - (audio.size(0) % self.segment_length)
                    audio = audio[:limit]

                nyq = 0.5 * self.get_sample_rate()
                normal_cutoff = self.cutoff / nyq
                b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)
                audio_compressed = signal.filtfilt(b, a, audio.numpy().copy()).copy()
                audio_compressed = torch.from_numpy(audio_compressed).float()

                audio = audio  # / MAX_WAV_VALUE
                audio_compressed = audio_compressed  # / MAX_WAV_VALUE

                audio = audio.unsqueeze(0)
                audio_compressed = audio_compressed.unsqueeze(0)
                spect = self.spectrogram(audio.squeeze(1)).squeeze(0)
                spect = torch.log(spect.clamp(min=1e-6))

                return audio, audio_compressed, spect

            index += 1

    def __len__(self):
        return len(self.files)

    @staticmethod
    def get_sample_rate():
        return 44100
