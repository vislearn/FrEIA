
from .base import Distribution

import torch


class NormalDistribution(Distribution):
    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        self.mean = mean
        self.var = var

    def sample(self, size: torch.Size = (), temperature: float = 1.0) -> torch.Tensor:
        ...

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        ...


class StandardNormalDistribution(NormalDistribution):
    def __init__(self, dim: int):
        mean = torch.zeros(dim)
        var = torch.ones(dim)
        super().__init__(mean, var)
