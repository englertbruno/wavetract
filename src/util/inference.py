#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------

import torch


@torch.no_grad()
def inference(x, model):
    model.eval()
    x = model(x)
    model.train()

    return x
