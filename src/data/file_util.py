#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------

import librosa
import torch
import torch.utils.data


def load_wav_to_torch(full_path, sample_rate):
    """
    Loads wavdata into torch array
    """
    data, sampling_rate = librosa.load(full_path, sr=sample_rate)  # read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


def load_mp3_to_torch(full_path, sample_rate):
    """
    Loads wavdata into torch array
    """
    data, sampling_rate = librosa.load(full_path, sr=sample_rate)
    return torch.from_numpy(data).float(), sampling_rate
