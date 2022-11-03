
from freia.core import Invertible

import torch


class Split(Invertible):
    def __init__(self, dim: int = 1):
        self.dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
