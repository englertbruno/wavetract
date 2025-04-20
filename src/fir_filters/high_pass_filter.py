#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------


import numpy as np


def high_pass_filter(f_c, f_samp, N):
    # algorithm FIR_high_pass filter
    # Input:
    # f_c, the cutoff frequency for the low-pass filter, in Hz
    # f_samp, sampling frequency of the audio signal to be filtered, in Hz
    # N, the order of the filter; assume N is odd
    #
    # Output:
    # h, a low-pass FIR filter in the form of an N-element array */

    # Normalize f_c and omega _c so that pi is equal to the Nyquist angular frequency
    f_c = f_c / f_samp
    omega_c = 2. * np.pi * f_c
    h = np.zeros(N, dtype=np.float64)
    for i in range(0, N):
        if i == N // 2 + 1:
            h[i] = 1 - 2. * f_c
        else:
            val = i - N // 2 - 1
            h[i] = -np.sin(omega_c * float(val)) / (np.pi * float(val))
    return h
