#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
from torch import nn

from src.model.layers import DoubleConv, Down, Up


class SpectNetConv(nn.Module):

    def __init__(self, embed_dim, n_blocks):
        super().__init__()

        self.d1 = Down(embed_dim, embed_dim)
        self.d2 = Down(embed_dim, embed_dim)
        self.d3 = Down(embed_dim, embed_dim)
        self.d4 = Down(embed_dim, embed_dim)

        self.u1 = Up(embed_dim, embed_dim)
        self.u2 = Up(embed_dim, embed_dim)
        self.u3 = Up(embed_dim, embed_dim)
        self.u4 = Up(embed_dim, embed_dim)

        self.mid = nn.Sequential(
            DoubleConv(embed_dim, embed_dim),
            DoubleConv(embed_dim, embed_dim),
        )

        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x_in):
        x1 = self.d1(x_in)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        x = self.mid(x4)

        x = self.u1(x) + x3
        x = self.u2(x) + x2
        x = self.u3(x) + x1
        x = self.u4(x)
        x = self.norm(x)
        return x
