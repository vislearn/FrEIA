
from freia.core import Invertible

import torch


WithJacobian = tuple[torch.Tensor, torch.Tensor]


class Transform(Invertible):
    def forward(self, x: torch.Tensor, *, condition: torch.Tensor, **kwargs) -> WithJacobian:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, **parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
