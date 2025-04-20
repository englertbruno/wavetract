#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
from typing import Callable

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from src.model.llm.vit_block import VITEncoderBlock


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class SpectNetTransformer(nn.Module):

    def __init__(self, embed_dim, n_blocks):
        super().__init__()
        seq_length = 128

        blocks_list = [
            VITEncoderBlock(embed_dim, embed_dim // 64)
            for _ in range(n_blocks)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_length, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        named_apply(init_weights_vit_timm, self)

    def forward(self, x):
        # print(x.shape)  torch.Size([1, 384, 128])
        x = x.permute(0, 2, 1) + self.pos_embed.to(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        return x


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
