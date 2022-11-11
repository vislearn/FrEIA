from functools import wraps

import torch

import torch.distributions

from typing import Dict


class Transform:
    def __init__(self):
        print(f"{self.__class__.__name__} __init__")

    def __call__(self, *args, **kwargs):
        print(f"{self.__class__.__name__} __call__")


from typing import Callable, Union

class Parameter:
    def __init__(self, count: Union[int, Callable[[Transform], int]]):
        self.count = count

    def initialize(self, transform: Transform):
        if isinstance(self.count, Callable):
            self.count = self.count(transform)

    def constrain(self, unconstrained: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Real(Parameter):
    def constrain(self, unconstrained: torch.Tensor) -> torch.Tensor:
        return unconstrained


class Positive(Parameter):
    def constrain(self, unconstrained: torch.Tensor) -> torch.Tensor:
        return torch.exp(unconstrained)


class Increasing(Parameter):
    def constrain(self, unconstrained: torch.Tensor) -> torch.Tensor:
        return unconstrained[:, 0] + torch.cumsum(torch.exp(unconstrained[:, 1:]), dim=1)



class Coupling(Transform):
    def __init__(self, split, transform, subnet, **parameters: Parameter):
        super().__init__()
        self.split = split
        self.transform = transform
        # TODO: 2 subnets? or just singular coupling?
        self.subnet = subnet

        test_subnet(...)
        try:
            self.subnet[-1].weight.data.zero_()
            self.subnet[-1].bias.data.zero_()
        except Exception:
            dummy_output = ...

            if not zero:
                warnings.warn(...)

        self._parameters = parameters

    @property
    def parameter_names(self):
        return self._parameters.keys()

    @property
    def parameter_counts(self):
        return [p.count for p in self._parameters.values()]

    def get_parameters(self, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        parameters = self.subnet(condition)
        parameters = torch.split(parameters, self.parameter_counts, dim=1)
        parameters = [p.constrain(u) for (p, u) in zip(self._parameters.keys(), parameters)]
        parameters = dict(zip(self.parameter_names, parameters))

        return parameters

    def transform_forward(self):
        pass

    def transform_inverse(self):
        pass

    def forward(self, x: torch.Tensor, rev: bool = False, jac: bool = True) -> torch.Tensor:
        x1, x2 = self.split.forward(x)
        parameters = self.get_parameters(x2)
        z1 = self.transform_forward(x1, **parameters)
        parameters = self.get_parameters(z1)
        z2 = self.transform.forward(x2, **parameters)

        z = self.split.inverse(z1, z2)

        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = self.split.forward(z)
        parameters = self.get_parameters(z1)
        x2 = self.transform.inverse(z2, **parameters)
        parameters = self.get_parameters(x2)
        x1 = self.transform.inverse(z1, **parameters)

        x = self.split.inverse(x1, x2)

        return x


from FrEIA.splits import EvenSplit


def parameterize(**parameters):
    def wrap(cls):
        @wraps(cls)
        def construct(*args, split=EvenSplit(), subnet_constructor, **kwargs):
            transform = cls(*args, **kwargs)
            for p in parameters.values():
                # initialize dynamic parameters
                p.initialize(transform)

            dims_in = ...
            dims_out = ...
            subnet = subnet_constructor(dims_in, dims_out)


            return Coupling(split=split, transform=transform, subnet=subnet, **parameters)
        return construct

    return wrap


# @parameterize(scale=Positive(1), shift=Real(1))
class AffineTransform(Coupling):
    def __init__(self):
        super().__init__(scale=Positive(1), shift=Real(1))
    def _forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, rev) -> torch.Tensor:
        return scale * x + shift

    def _inverse(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return (x - shift) / scale


@parameterize(x_edges=Increasing(lambda t: t.bins), y_edges=Increasing(lambda t: t.bins), deltas=Increasing(lambda t: t.bins - 1))
class RQSpline(Transform):
    def __init__(self, bins: int):
        super().__init__(x_edges=Increasing(bins))
        self.bins = bins

    def forward(self, x: torch.Tensor, x_edges: torch.Tensor, y_edges: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*x.shape)

    def blub(self):
        pass

    def inverse(self):
        pass


def subnet_constructor(dims_in, dims_out):
    print("subnet_constructor")
    def subnet(x):
        return torch.zeros(x.shape[0], dims_out)

    return subnet


t = AffineTransform(subnet_constructor=subnet_constructor)
t.blub()
print(type(t.transform.blub()))

x = None

t(x)


t = RQSpline(bins=8, subnet_constructor=subnet_constructor)

t(x)

print(t.parameter_counts)

