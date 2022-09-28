from typing import Tuple

import torch

from .binned import BinnedSpline


class LinearSpline(BinnedSpline):
    def __init__(self, *args, bins: int = 10, **kwargs):
        #       parameter                           constraints             count
        # 1.    the domain half-width B             positive                1
        # 2.    the relative width of each bin      positive, sum to 1      #bins
        # 3.    the relative height of each bin     positive, sum to 1      #bins
        super().__init__(*args, **kwargs, bins=bins, parameter_counts=[1, bins, bins])

    def spline1(self,
                x: torch.Tensor,
                left: torch.Tensor,
                right: torch.Tensor,
                bottom: torch.Tensor,
                top: torch.Tensor,
                *params: torch.Tensor,
                rev: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        return linear_spline(x, left, right, bottom, top, rev=rev)

    def spline2(self,
                x: torch.Tensor,
                left: torch.Tensor,
                right: torch.Tensor,
                bottom: torch.Tensor,
                top: torch.Tensor,
                *params: torch.Tensor,
                rev: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        return linear_spline(x, left, right, bottom, top, rev=rev)


def linear_spline(x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, bottom: torch.Tensor, top: torch.Tensor,
                  rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a linear spline, connecting each knot with a straight line
    Args:
        x: input tensor
        left: left bin edges for inputs inside the spline domain
        right: same as left
        bottom: same as left
        top: same as left
        rev: whether to run in reverse

    Returns:
        splined output tensor
        log jacobian matrix entries
    """
    dx = right - left
    dy = top - bottom

    a = dy / dx
    b = bottom - a * left

    if not rev:
        out = a * x + b
    else:
        out = (x - b) / a

    log_jac = torch.log(dy) - torch.log(dx)

    return out, log_jac
