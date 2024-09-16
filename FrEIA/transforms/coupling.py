
from .base import Transform
from freia.splits import EvenSplit

import torch
import torch.nn as nn


class Spline(Transform):
    def __init__(self, affine, inner_spline):
        ...

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor, **kwargs) -> WithJacobian:
        x[out] = affine(x[out])
        x[out] = inner_spline(x[out])


class Spline(CouplingTransform):
    def _forward(self):
        x[in] = self._spline(...)
        x[out] = self._affine(...)



class CouplingTransform(Transform):
    def __init__(self, transform1, transform2, subnet_constructor, split=EvenSplit(dim=1)):
        self.split = split
        self.subnet1 = subnet_constructor(...)
        self.subnet2 = subnet_constructor(...)

    def split_parameters(self, parameters: torch.Tensor) -> dict:
        pc = self.parameter_counts
        parameters = torch.split(parameters, list(pc.values()), dim=1)

        return dict(zip(pc.keys(), parameters))

    def transform_parameters(self, parameters: dict[torch.Tensor]) -> None:
        pass

    def get_parameters(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def get_parameters(self, u: torch.Tensor, subnet: nn.Module) -> dict:

        parameters = subnet(u)
        parameters = self.split_parameters(parameters)
        should_be_none = self.transform_parameters(**parameters)
        if should_be_none is not None:
            warnings.warn(...)

        return parameters


    def forward(self, x: torch.Tensor, **parameters: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.split.forward(x)



        parameters = self.get_parameters(u=x2, subnet=self.subnet1)
        z1, logdet1 = self.transform1.forward(x1, **parameters)
        parameters = self.get_parameters(u=z1, subnet=self.subnet2)
        z2, logdet2 = self.transform2(x2, **parameters)

        z = self.split.inverse(z1, z2)
        logdet = logdet1 + logdet2

        return z, logdet





my_single_coupling = CouplingTransform(transform1=AffineTransform(...), transform2=None)
