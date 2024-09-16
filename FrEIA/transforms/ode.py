
from .base import Transform

import torch

from scipy.ode import solve_ode



class Parameterized(nn.Module):
    def __init__(self, *, subnet_constructor, parameter_counts):
        super().__init__()
        self.subnet = ...
        self.parameter_counts = ...
        self.transform = transform_cls

    def __call__(self, *args, **kwargs):
        self.transform = transform_cls(*args, **kwargs)

        return self

    def forward(self):
        parameters = self.subnet(...)
        return self.transform(x, parameters)


@Parameterized
class ODETransform(Transform):
    def __init__(self, integration_steps: int = 10):
        super().__init__()
        self.integration_steps = integration_steps

    def forward(self, x: torch.Tensor, **parameters) -> tuple[torch.Tensor, torch.Tensor]:
        return euler(x, v, dt)

        # ode integration
        dt = 1 / self.integration_steps
        for _ in range(self.integration_steps):
            parameters = self.get_parameters()
            v = parameters["v"]
            x = euler(x, v, dt)

        return x

ODETransform = Parameterized(ODETransform)





ode = ODETransform()




def euler(x, v, dt):
    return x + v * dt


