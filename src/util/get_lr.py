#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------

import math


def get_lr(i, warmup_iters, max_iters, base_lr, final_lr):
    if i < warmup_iters:
        return base_lr * i / warmup_iters
    if i > max_iters:
        return final_lr
    decay_ratio = (i - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return final_lr + coeff * (base_lr - final_lr)
