#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------from typing import Callable, Optional

from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            drop: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ResMlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            drop: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = Mlp(in_features, hidden_features, out_features, act_layer, drop, bias)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mlp(self.norm(x))
        return x
