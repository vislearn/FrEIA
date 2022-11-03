
from .base import Transform

import torch

from .coupling import CouplingTransform


class AffineTransform(CouplingTransform):

    def __init__(self):
        parameter_counts = {...}
        super().__init__(parameter_counts=parameter_counts)

    def transform_parameters(self, **parameters):
        parameters["a"] = torch.exp(parameters["a"])

    def _forward(self, x: torch.Tensor, **parameters) -> torch.Tensor:
        parameters = self.get_parameters()
        a, b = parameters["a"], parameters["b"]
        return a * x + b, torch.log(a)

    def _inverse(self, z: torch.Tensor, **parameters) -> torch.Tensor:
        a, b = parameters["a"], parameters["b"]
        return (z - b) / a, -torch.log(a)
