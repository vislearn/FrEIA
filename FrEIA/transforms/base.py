
from freia.core import Invertible

import torch


WithJacobian = tuple[torch.Tensor, torch.Tensor]



class Transform(Invertible):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError




@Parameterized(scale=1, shift=1)
class AffineTransform(Transform):
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return scale * x + shift

    def inverse(self, z: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return (z - shift) / scale


# class SplineTransform(Transform):
#     def forward(self, x: torch.Tensor, edges: torch.Tensor):
#         assert edges.shape == (..., self.bins)
#         pass


class Parameterized:
    def __init__(self, **parameter_counts):
        self.parameter_counts = parameter_counts

    def __call__(self, cls):

        cls.forward = forward
        cls.inverse = inverse



