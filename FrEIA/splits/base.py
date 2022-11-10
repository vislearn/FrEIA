
from FrEIA.core import Invertible

from typing import Tuple

import torch


class Split(Invertible):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
