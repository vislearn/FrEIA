from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from itertools import chain

from FrEIA.modules.coupling_layers import _BaseCouplingBlock
from FrEIA import utils


class BinnedSpline(_BaseCouplingBlock):
    """
    Base Class for Splines
    Implements input-binning, where bin knots are jointly predicted along with spline parameters
    by a non-invertible coupling subnetwork
    """
    def __init__(self, dims_in, dims_c=None, subnet_constructor: callable = None, split_len: Union[float, int] = 0.5,
                 bins: int = 10, parameter_counts: List[int] = None):
        if dims_c is None:
            dims_c = []
        if parameter_counts is None:
            parameter_counts = [1, bins, bins]

        super().__init__(dims_in, dims_c, clamp=0.0, clamp_activation=lambda u: u, split_len=split_len)

        num_params = sum(parameter_counts)
        self.subnet1 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * num_params)
        self.subnet2 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * num_params)

        self.bins = bins
        self.parameter_counts = parameter_counts

    def spline1(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, bottom: torch.Tensor, top: torch.Tensor,
                parameters: List[torch.Tensor], rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the actual spline. Callees need to implement this method.
        Args:
            x: Inputs within the spline domain
            left: Bin edge to the left of each input
            right: Bin edge to the right of each input
            bottom: Bin edge below each input
            top: Bin edge above each input
            parameters: Additional spline parameters
            rev: whether to run in reverse

        Returns:
            splined output tensor
            log jacobian matrix entries for splined inputs
        """
        raise NotImplementedError

    def spline2(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, bottom: torch.Tensor, top: torch.Tensor,
                parameters: List[torch.Tensor], rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def split_parameters(self, parameters: torch.Tensor, split_len: int) -> List[torch.Tensor]:
        """
        Split network output into semantic parameters, as given by self.parameter_counts
        """
        parameters = parameters.movedim(1, -1)
        parameters = parameters.reshape(*parameters.shape[:-1], split_len, -1)
        return list(torch.split(parameters, self.parameter_counts, dim=-1))

    def constrain_parameters(self, parameters: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Constrain Parameters to meet certain conditions (e.g. positivity)
        """
        domain_width, bin_widths, bin_heights, *parameters = parameters

        # constrain the domain width to positive values
        # use a shifted softplus
        shift = np.log(np.e - 1)
        domain_width = F.softplus(domain_width + shift)

        # bin widths must be positive and sum to 1
        bin_widths = F.softmax(bin_widths, dim=-1)

        # same for the bin heights
        bin_heights = F.softmax(bin_heights, dim=-1)

        return [domain_width, bin_widths, bin_heights, *parameters]

    def bin_parameters(self, x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, parameters: List[torch.Tensor],
                       rev: bool = False) -> List[torch.Tensor]:
        """
        Sort Parameters into bins, returning left and right edges for each.
        Inputs that are outside the spline domain are not binned.
        Also returns which parameters are inside the spline domain.
        """
        # determine which inputs are inside the spline domain
        # i.e. between the first and last bin
        if not rev:
            inside = (xs[..., 0] < x) & (x <= xs[..., -1])
        else:
            y = x
            inside = (ys[..., 0] < y) & (y <= ys[..., -1])

        #
        x = x[inside]
        xs = xs[inside]
        ys = ys[inside]
        parameters = [p[inside] for p in parameters]

        # find upper and lower bin edge indices
        if not rev:
            upper = torch.searchsorted(xs, x[..., None])
            lower = upper - 1
        else:
            y = x
            upper = torch.searchsorted(ys, y[..., None])
            lower = upper - 1

        left_edge = torch.gather(xs, dim=-1, index=lower).squeeze(-1)
        right_edge = torch.gather(xs, dim=-1, index=upper).squeeze(-1)
        bottom_edge = torch.gather(ys, dim=-1, index=lower).squeeze(-1)
        top_edge = torch.gather(ys, dim=-1, index=upper).squeeze(-1)

        pleft = [torch.gather(p, dim=-1, index=lower).squeeze(-1) for p in parameters]
        pright = [torch.gather(p, dim=-1, index=upper).squeeze(-1) for p in parameters]

        # interleave pleft and pright
        parameters = list(chain.from_iterable(zip(pleft, pright)))

        return [inside, left_edge, right_edge, bottom_edge, top_edge, *parameters]

    def _coupling1(self, x1: torch.Tensor, u2: torch.Tensor, rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The full coupling consists of:
        - Querying Parameters from the subnetwork
        - Transforming the unconstrained parameters into bin parameters and spline parameters
        - Performing the actual spline
        """
        parameters = self.subnet1(u2)
        parameters = self.split_parameters(parameters, self.split_len1)
        parameters = self.constrain_parameters(parameters)

        domain_width, bin_widths, bin_heights, *parameters = parameters

        xs, ys = make_knots(domain_width, bin_widths, bin_heights)

        inside, left, right, bottom, top, *parameters = self.bin_parameters(x1, xs, ys, parameters, rev=rev)

        return binned_spline(
            x1,
            inside=inside,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            spline=self.spline1,
            spline_parameters=parameters,
            rev=rev,
        )

    def _coupling2(self, x2: torch.Tensor, u1: torch.Tensor, rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        parameters = self.subnet2(u1)
        parameters = self.split_parameters(parameters, self.split_len2)
        parameters = self.constrain_parameters(parameters)

        domain_width, bin_widths, bin_heights, *parameters = parameters

        xs, ys = make_knots(domain_width, bin_widths, bin_heights)

        inside, left, right, bottom, top, *parameters = self.bin_parameters(x2, xs, ys, parameters, rev=rev)

        return binned_spline(
            x2,
            inside=inside,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            spline=self.spline2,
            spline_parameters=parameters,
            rev=rev,
        )


def binned_spline(x: torch.Tensor, *, inside: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
                  bottom: torch.Tensor, top: torch.Tensor, spline: callable,
                  spline_parameters: List[torch.Tensor], rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a binned spline with identity tails
    Args:
        x: input tensor
        inside: which inputs are inside the spline domain
        left: left bin edges for inputs inside the spline domain
        right: same as left
        bottom: same as left
        top: same as left
        spline: the spline function
        spline_parameters: extra parameters for the spline function
        rev: whether to run in reverse

    Returns:
        splined output tensor
        log jacobian determinant
    """
    # identity tails
    out = torch.clone(x)
    log_jac = out.new_zeros(out.shape)

    # print(f"{out.shape=}, {log_jac.shape=}, {inside.shape=}, {spline_out.shape=}, {spline_log_jac.shape=}, {x[inside].shape=}")

    # overwrite inside with spline
    out[inside], log_jac[inside] = spline(x[inside], left, right, bottom, top, spline_parameters, rev=rev)

    # the determinant is given by the sum
    log_jac_det = utils.sum_except_batch(log_jac)

    if rev:
        log_jac_det = -log_jac_det

    return out, log_jac_det


def make_knots(domain_width: torch.Tensor,
               bin_widths: torch.Tensor,
               bin_heights: torch.Tensor,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find bin knots (x and y coordinates) from constrained parameters
    Args:
        domain_width: half-width of the zero-centered spline box
        bin_widths: relative widths of each bin
        bin_heights: relative heights of each bin

    Returns:
        tuple containing bin knot x and y coordinates
    """

    xs = torch.cumsum(bin_widths, dim=-1)
    pad = xs.new_zeros((*xs.shape[:-1], 1))
    xs = torch.cat((pad, xs), dim=-1)
    xs = 2 * domain_width * xs - domain_width

    ys = torch.cumsum(bin_heights, dim=-1)
    pad = ys.new_zeros((*ys.shape[:-1], 1))
    ys = torch.cat((pad, ys), dim=-1)
    ys = 2 * domain_width * ys - domain_width

    return xs, ys
