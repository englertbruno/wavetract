#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from pylab import specgram

from src.data.ljspeech_dataset import MAX_WAV_VALUE


def viz(audio_compressed, noise, uncompressed_audio, audio, iter, sample_rate, out_path):
    specgrams = []
    for i, data in enumerate([audio_compressed, noise, uncompressed_audio, audio]):
        data = (data[0, 0, :].cpu().detach().numpy() * MAX_WAV_VALUE).astype('int16')
        spec, freqs, t, im = specgram(data, NFFT=512, Fs=sample_rate, noverlap=480)
        spec = np.flipud(np.log(np.abs(spec) + 1e-7))
        specgrams.append((spec, freqs, t, im))

    spec1, freqs, t, im = specgrams[-2]
    spec2, freqs, t, im = specgrams[-1]

    spect_diff = (spec1 - spec2) ** 2

    pad_xextent = (128. - 64.) / sample_rate / 2.
    xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent

    xmin, xmax = xextent
    extent = xmin, xmax, freqs[0], freqs[-1]

    fontsize = 30
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(20, 20))
    i = 0
    ax[i].imshow(specgrams[0][0], extent=extent)
    ax[i].axis('auto')
    ax[i].set_title("compressed", fontsize=fontsize)

    i += 1
    ax[i].imshow(specgrams[1][0], extent=extent)
    ax[i].axis('auto')
    ax[i].set_title("noise", fontsize=fontsize)

    i += 1
    ax[i].imshow(specgrams[2][0], extent=extent)
    ax[i].axis('auto')
    ax[i].set_title("uncompressed", fontsize=fontsize)

    i += 1
    ax[i].imshow(spect_diff, extent=extent)
    ax[i].axis('auto')
    ax[i].set_title("diff", fontsize=fontsize)

    i += 1
    ax[i].imshow(specgrams[3][0], extent=extent)
    ax[i].axis('auto')
    ax[i].set_title("original", fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close('all')
