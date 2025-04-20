#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------

import torch


class Spectrogram(object):
    """Create a spectrogram from a raw audio signal
    Args:
        n_fft (int, optional): size of fft, creates n_fft // 2 + 1 bins
        ws (int): window size. default: n_fft
        hop (int, optional): length of hop between STFT windows. default: ws // 2
        pad (int): two sided padding of signal
        window (torch windowing function): default: torch.hann_window
        power (int > 0 ) : Exponent for the magnitude spectrogram,
                        e.g., 1 for energy, 2 for power, etc.
        normalize (bool) : whether to normalize by magnitude after stft
        wkwargs (dict, optional): arguments for window function
    """

    def __init__(self, n_fft=400, ws=None, hop=None,
                 pad=0, window=torch.hann_window,
                 power=2, normalize=False, wkwargs=None, center=True):
        self.n_fft = n_fft
        # number of fft bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.ws = ws if ws is not None else n_fft
        self.hop = hop if hop is not None else self.ws // 2
        self.window = window(self.ws) if wkwargs is None else window(self.ws, **wkwargs)
        self.pad = pad
        self.power = power
        self.normalize = normalize
        self.wkwargs = wkwargs
        self.center = center

    def __call__(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (c, n)
        Returns:
            spec_f (Tensor): channels x hops x n_fft (c, l, f), where channels
                is unchanged, hops is the number of hops, and n_fft is the
                number of fourier bins, which should be the window size divided
                by 2 plus 1.
        """
        assert sig.dim() == 2

        if self.pad > 0:
            with torch.no_grad():
                sig = torch.nn.functional.pad(sig, (self.pad, self.pad), "constant")
        self.window = self.window.to(sig.device)

        # default values are consistent with librosa.core.spectrum._spectrogram
        spec_f = torch.stft(sig, self.n_fft, self.hop, self.ws,
                            self.window, center=self.center,
                            normalized=False, onesided=True,
                            pad_mode='reflect', return_complex=False).transpose(1, 2)
        if self.normalize:
            spec_f /= self.window.pow(2).sum().sqrt()
        spec_f = spec_f.pow(self.power).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
        return spec_f
