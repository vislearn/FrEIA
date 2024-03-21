
from .base import Transform

import torch


class IdentityTransform(Transform):
    def forward(self, x: torch.Tensor, **parameters: torch.Tensor) -> WithJacobian:
        return x, 0

    def inverse(self, z: torch.Tensor, **parameters: torch.Tensor) -> WithJacobian:
        return z, 0
