import torch
from torch.distributions import Independent, Normal


class StandardNormalDistribution(Independent):
    def __init__(self, *event_shape: int, device=None, dtype=None):
        loc = torch.tensor(0., device=device, dtype=dtype).repeat(event_shape)
        scale = torch.tensor(1., device=device, dtype=dtype).repeat(event_shape)

        super().__init__(Normal(loc, scale), len(event_shape))
