import torch
from torch.distributions import Normal, Independent


class NormalDistribution(Independent):
    def __init__(self,
                 loc: torch.Tensor,
                 scale: torch.Tensor,
                 event_dim_count: int):
        super().__init__(Normal(loc, scale), event_dim_count)


class StandardNormalDistribution(NormalDistribution):
    def __init__(self, *event_shape: int):
        super().__init__(torch.tensor(0.).repeat(event_shape),
                         torch.tensor(1.).repeat(event_shape), len(event_shape))
