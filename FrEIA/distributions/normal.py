import torch
from torch.distributions import Independent, Normal


class StandardNormalDistribution(Independent):
    def __init__(self, *event_shape: int, device=None, dtype=None, validate_args=True):
        loc = torch.tensor(0., device=device, dtype=dtype).repeat(event_shape)
        scale = torch.tensor(1., device=device, dtype=dtype).repeat(event_shape)

        super().__init__(
            Normal(loc, scale, validate_args=validate_args),
            len(event_shape),
            validate_args=validate_args
        )
