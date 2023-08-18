from typing import Dict, Tuple, Callable, Iterable, List

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .binned import BinnedSplineBase, BinnedSpline


class RationalQuadraticSpline(BinnedSpline):
    def __init__(self, *args, bins: int = 10, **kwargs):
        #       parameter                                       constraints             count
        # 1.    the derivative at the edge of each inner bin    positive                #bins - 1
        super().__init__(*args, bins=bins, parameter_counts={"deltas": bins - 1}, **kwargs)

    def constrain_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        parameters = super().constrain_parameters(parameters)
        # we additionally want positive derivatives to preserve monotonicity
        # the derivative must also match the tails at the spline boundaries
        deltas = parameters["deltas"]

        # shifted softplus such that network output 0 -> delta = scale
        shift = np.log(np.e - 1)
        deltas = F.softplus(deltas + shift)

        # boundary condition: derivative is equal to affine scale at spline boundaries
        scale = torch.sum(parameters["heights"], dim=-1, keepdim=True) / torch.sum(parameters["widths"], dim=-1, keepdim=True)
        scale = scale.expand(*scale.shape[:-1], 2)

        deltas = torch.cat((deltas, scale), dim=-1).roll(1, dims=-1)

        parameters["deltas"] = deltas

        return parameters

    def _spline1(self, x: torch.Tensor, parameters: Dict[str, torch.Tensor], rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        left, right, bottom, top = parameters["left"], parameters["right"], parameters["bottom"], parameters["top"]
        deltas_left, deltas_right = parameters["deltas_left"], parameters["deltas_right"]
        return rational_quadratic_spline(x, left, right, bottom, top, deltas_left, deltas_right, rev=rev)

    def _spline2(self, x: torch.Tensor, parameters: Dict[str, torch.Tensor], rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        left, right, bottom, top = parameters["left"], parameters["right"], parameters["bottom"], parameters["top"]
        deltas_left, deltas_right = parameters["deltas_left"], parameters["deltas_right"]
        return rational_quadratic_spline(x, left, right, bottom, top, deltas_left, deltas_right, rev=rev)


class ElementwiseRationalQuadraticSpline(BinnedSplineBase):
    def __init__(self, dims_in, dims_c=[], subnet_constructor: Callable = None, 
                 bins: int = 10, **kwargs) -> None:
        super().__init__(dims_in, dims_c, bins=bins, parameter_counts={"deltas": bins - 1}, **kwargs)

        self.channels = dims_in[0][0]
        self.condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])
        self.conditional = len(dims_c) > 0

        num_params = sum(self.parameter_counts.values())


        if self.conditional:
            self.subnet = subnet_constructor(self.condition_length, self.channels * num_params)
        else:
            self.spline_parameters = nn.Parameter(torch.zeros(self.channels * num_params, *dims_in[0][1:]))


    def constrain_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        parameters = super().constrain_parameters(parameters)
        # we additionally want positive derivatives to preserve monotonicity
        # the derivative must also match the tails at the spline boundaries
        deltas = parameters["deltas"]

        # shifted softplus such that network output 0 -> delta = scale
        shift = np.log(np.e - 1)
        deltas = F.softplus(deltas + shift)

        # boundary condition: derivative is equal to affine scale at spline boundaries
        scale = torch.sum(parameters["heights"], dim=-1, keepdim=True) / torch.sum(parameters["widths"], dim=-1, keepdim=True)
        scale = scale.expand(*scale.shape[:-1], 2)

        deltas = torch.cat((deltas, scale), dim=-1).roll(1, dims=-1)

        parameters["deltas"] = deltas

        return parameters

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        return input_dims

    def forward(self, x_or_z: Iterable[torch.Tensor], c: Iterable[torch.Tensor] = None,
                rev: bool = False, jac: bool = True) \
            -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        if self.conditional:
            parameters = self.subnet(torch.cat(c, dim=1).float())
        else:
            parameters = self.spline_parameters.unsqueeze(0).repeat_interleave(x_or_z[0].shape[0], dim=0)
        parameters = self.split_parameters(parameters, self.channels)
        parameters = self.constrain_parameters(parameters)

        y, jac = self.binned_spline(x=x_or_z[0], parameters=parameters, spline=self.spline, rev=rev)
        return (y,), jac

    def spline(self, x: torch.Tensor, parameters: Dict[str, torch.Tensor], rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        left, right, bottom, top = parameters["left"], parameters["right"], parameters["bottom"], parameters["top"]
        deltas_left, deltas_right = parameters["deltas_left"], parameters["deltas_right"]
        return rational_quadratic_spline(x, left, right, bottom, top, deltas_left, deltas_right, rev=rev)

def rational_quadratic_spline(x: torch.Tensor,
                              left: torch.Tensor,
                              right: torch.Tensor,
                              bottom: torch.Tensor,
                              top: torch.Tensor,
                              deltas_left: torch.Tensor,
                              deltas_right: torch.Tensor,
                              rev: bool = False,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rational-quadratic spline with the algorithm described in arXiv:1906.04032
    Forward output is defined for each bin by the fraction
    ..math::
        z = f(x) = \\frac{ \\beta_0 + \\beta_1 x + \\beta_2 x^2 }{ 1 + \\beta_3 x + \\beta_4 x^2 }

    where the \\beta are constrained to yield a smooth function over all bins
    Args:
        x: input tensor
        left: left bin edges for inputs inside the spline domain
        right: same as left
        bottom: same as left
        top: same as left
        deltas_left: spline derivatives at the left and bottom bin edge
        deltas_right: spline derivatives at the right and top bin edge
        rev: whether to run in reverse

    Returns:
        splined output tensor
        log jacobian matrix entries
    """
    # rename variables to match the paper:
    # xk means $x_k$ and xkp means $x_{k+1}$
    xk = left
    xkp = right
    yk = bottom
    ykp = top
    dk = deltas_left
    dkp = deltas_right

    # define some commonly used values
    dx = xkp - xk
    dy = ykp - yk
    sk = dy / dx

    if not rev:
        xi = (x - xk) / dx

        # Eq 4 in the paper
        numerator = dy * (sk * xi ** 2 + dk * xi * (1 - xi))
        denominator = sk + (dkp + dk - 2 * sk) * xi * (1 - xi)
        out = yk + numerator / denominator
    else:
        y = x
        # Eq 6-8 in the paper
        a = dy * (sk - dk) + (y - yk) * (dkp + dk - 2 * sk)
        b = dy * dk - (y - yk) * (dkp + dk - 2 * sk)
        c = -sk * (y - yk)

        # Eq 29 in the appendix of the paper
        discriminant = b ** 2 - 4 * a * c
        if not torch.all(discriminant >= 0):
            raise(RuntimeError(f"Discriminant must be positive, but is violated by {torch.min(discriminant)}"))

        xi = 2 * c / (-b - torch.sqrt(discriminant))

        out = xi * dx + xk

    # Eq 5 in the paper
    numerator = sk ** 2 * (dkp * xi ** 2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
    denominator = (sk + (dkp + dk - 2 * sk) * xi * (1 - xi)) ** 2
    log_jac = torch.log(numerator) - torch.log(denominator)

    return out, log_jac


