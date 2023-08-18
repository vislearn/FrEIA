from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from itertools import chain

from FrEIA.modules.coupling_layers import _BaseCouplingBlock
from FrEIA.modules.base import InvertibleModule
from FrEIA import utils

class BinnedSpline(_BaseCouplingBlock):
    def __init__(self, dims_in, dims_c=None, subnet_constructor: callable = None, 
                 split_len: Union[float, int] = 0.5, **kwargs) -> None:
        if dims_c is None:
            dims_c = []

        super().__init__(dims_in, dims_c, clamp=0.0, clamp_activation=lambda u: u, split_len=split_len)

        
        self.spline_base = BinnedSplineBase(dims_in, dims_c, **kwargs)
        
        num_params = sum(self.spline_base.parameter_counts.values())
        self.subnet1 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * num_params)
        self.subnet2 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * num_params)

    def _spline1(self, x1: torch.Tensor, parameters: Dict[str, torch.Tensor], rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _spline2(self, x2: torch.Tensor, parameters: Dict[str, torch.Tensor], rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _coupling1(self, x1: torch.Tensor, u2: torch.Tensor, rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The full coupling consists of:
        1. Querying the parameter tensor from the subnetwork
        2. Splitting this tensor into the semantic parameters
        3. Constraining the parameters
        4. Performing the actual spline for each bin, given the parameters
        """
        parameters = self.subnet1(u2)
        parameters = self.spline_base.split_parameters(parameters, self.split_len1)
        parameters = self.constrain_parameters(parameters)

        return self.spline_base.binned_spline(x=x1, parameters=parameters, spline=self._spline1, rev=rev)

    def _coupling2(self, x2: torch.Tensor, u1: torch.Tensor, rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        parameters = self.subnet2(u1)
        parameters = self.spline_base.split_parameters(parameters, self.split_len2)
        parameters = self.constrain_parameters(parameters)

        return self.spline_base.binned_spline(x=x2, parameters=parameters, spline=self._spline2, rev=rev)

    def constrain_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.spline_base.constrain_parameters(parameters)

class BinnedSplineBase(InvertibleModule):
    """
    Base Class for Splines
    Implements input-binning, where bin knots are jointly predicted along with spline parameters
    by a non-invertible coupling subnetwork
    """

    def __init__(self, dims_in, dims_c=None, bins: int = 10, parameter_counts: Dict[str, int] = None, 
                 min_bin_sizes: Tuple[float] = (0.1, 0.1), default_domain: Tuple[float] = (-3.0, 3.0, -3.0, 3.0),
                 identity_tails: bool = False, domain_clamping: float = None) -> None:
        """
        Args:
            bins: number of bins to use
            parameter_counts: dictionary containing (parameter_name, parameter_counts)
                the counts are used to split the network outputs
            min_bin_sizes: tuple of (min_x_size, min_y_size)
                bins are scaled such that they never fall below this size
            default_domain: tuple of (left, right, bottom, top) default spline domain values
                these values will be used as the starting domain (when the network outputs zero)
            identity_tails: whether to use identity tails for the spline
            domain_clamping: clamping value for the domain, if float,
                clamp spline width and height to (-domain_clamping, domain_clamping)
        """
        if dims_c is None:
            dims_c = []
        if parameter_counts is None:
            parameter_counts = {}

        super().__init__(dims_in, dims_c)

        assert bins >= 1, "need at least one bin"
        assert all(s >= 0 for s in min_bin_sizes), "minimum bin size cannot be negative"
        assert default_domain[1] > default_domain[0], "x domain must be increasing"
        assert default_domain[3] > default_domain[2], "y domain must be increasing"
        assert default_domain[1] - default_domain[0] >= min_bin_sizes[0] * bins, \
        "{bins} bins of size {min_bin_sizes[0]} are too large for domain {default_domain[0]} to {default_domain[1]}"
        assert default_domain[3] - default_domain[2] >= min_bin_sizes[1] * bins, \
        "{bins} bins of size {min_bin_sizes[1]} are too large for domain {default_domain[2]} to {default_domain[3]}"

        self.register_buffer("bins", torch.tensor(bins, dtype=torch.int32))
        self.register_buffer("min_bin_sizes", torch.as_tensor(min_bin_sizes, dtype=torch.float32))
        self.register_buffer("default_domain", torch.as_tensor(default_domain, dtype=torch.float32))
        self.register_buffer("identity_tails", torch.tensor(identity_tails, dtype=torch.bool))
        self.register_buffer("default_width", torch.as_tensor(default_domain[1] - default_domain[0], dtype=torch.float32))

        self.domain_clamping = domain_clamping

        # The default parameters are
        #       parameter                                       constraints             count
        # 1.    the leftmost bin edge                           -                       1
        # 2.    the lowermost bin edge                          -                       1
        # 3.    the widths of each bin                          positive                #bins
        # 4.    the heights of each bin                         positive                #bins
        default_parameter_counts = dict(
            widths=bins,
            heights=bins,
        )
        if not identity_tails:
            default_parameter_counts["bottom"] = 1
            default_parameter_counts["left"] = 1
        else:
            default_parameter_counts["total_width"] = 1
        # merge parameter counts with child classes
        self.parameter_counts = {**default_parameter_counts, **parameter_counts}


    def split_parameters(self, parameters: torch.Tensor, split_len: int) -> Dict[str, torch.Tensor]:
        """
        Split parameter tensor into semantic parameters, as given by self.parameter_counts
        """
        keys = list(self.parameter_counts.keys())
        lengths = list(self.parameter_counts.values())

        parameters = parameters.movedim(1, -1)
        parameters = parameters.reshape(*parameters.shape[:-1], split_len, -1)

        values = list(torch.split(parameters, lengths, dim=-1))

        return dict(zip(keys, values))

    def clamp_domain(self, domain: torch.Tensor) -> torch.Tensor:
        """
        Clamp domain to the a size between (-domain_clamping, domain_clamping)
        """
        if self.domain_clamping is None:
            return domain
        else:
            return self.domain_clamping * torch.tanh(domain / self.domain_clamping)

    def constrain_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Constrain Parameters to meet certain conditions (e.g. positivity)
        """
        # we constrain the widths and heights to be positive with a softplus
        # furthermore, to allow minimum bin widths, we add this outside the softplus
        # we also want to use the default domain when the network predicts zeros, so
        # shift the softplus such that this is true, even with nonzero minimum bin sizes.
        if self.identity_tails:
            total_width = parameters["total_width"]
            shift = np.log(np.e - 1)
            total_width = self.default_width * F.softplus(total_width + shift)
            total_width = self.clamp_domain(total_width)
            parameters["left"] = -total_width / 2
            parameters["bottom"] = -total_width / 2

            parameters["widths"] = total_width * F.softmax(parameters["widths"], dim=-1)
            parameters["heights"] = total_width * F.softmax(parameters["heights"], dim=-1)

        else:
            parameters["left"] = parameters["left"] + self.default_domain[0]
            parameters["bottom"] = parameters["bottom"] + self.default_domain[2]

            default_bin_width = (self.default_domain[1] - self.default_domain[0]) / self.bins
            default_bin_height = (self.default_domain[3] - self.default_domain[2]) / self.bins

            xshift = torch.log(torch.exp(default_bin_width - self.min_bin_sizes[0]) - 1)
            yshift = torch.log(torch.exp(default_bin_height - self.min_bin_sizes[1]) - 1)

            parameters["widths"] = self.min_bin_sizes[0] + F.softplus(parameters["widths"] + xshift)
            parameters["heights"] = self.min_bin_sizes[1] + F.softplus(parameters["heights"] + yshift)

            domain_width = torch.sum(parameters["widths"], dim=-1, keepdim=True)
            domain_height = torch.sum(parameters["heights"], dim=-1, keepdim=True)
            width_resize = self.clamp_domain(domain_width) / domain_width
            height_resize = self.clamp_domain(domain_height) / domain_height

            parameters["widths"] = parameters["widths"] * width_resize
            parameters["heights"] = parameters["heights"] * height_resize
            parameters["left"] = parameters["left"] * width_resize
            parameters["bottom"] = parameters["bottom"] * height_resize

        return parameters

    def binned_spline(self, x: torch.Tensor, parameters: Dict[str, torch.Tensor], spline: callable, rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the spline for given bin and spline parameters
        """
        x = x.movedim(1, -1)

        # find bin knots
        knot_x = parameters["left"] + torch.cumsum(parameters["widths"], dim=-1)
        knot_y = parameters["bottom"] + torch.cumsum(parameters["heights"], dim=-1)

        # concatenate leftmost edge
        knot_x = torch.cat((parameters["left"], knot_x), dim=-1)
        knot_y = torch.cat((parameters["bottom"], knot_y), dim=-1)


        # find spline mask
        if not rev:
            inside = (knot_x[..., 0] < x) & (x <= knot_x[..., -1])
        else:
            y = x
            inside = (knot_y[..., 0] < y) & (y <= knot_y[..., -1])

        knot_x = knot_x[inside]
        knot_y = knot_y[inside]

        x_in = x[inside]
        x_out = x[~inside]

        scale = torch.sum(parameters["heights"], dim=-1, keepdim=True) / torch.sum(parameters["widths"], dim=-1, keepdim=True)
        shift = parameters["bottom"] - scale * parameters["left"]

        scale = scale[~inside].squeeze(-1)
        shift = shift[~inside].squeeze(-1)

        # find bin edge indices
        if not rev:
            upper = torch.searchsorted(knot_x, x_in[..., None])
        else:
            y_in = x_in
            upper = torch.searchsorted(knot_y, y_in[..., None])
        lower = upper - 1

        spline_parameters = dict()

        # gather bin edges from indices
        spline_parameters["left"] = torch.gather(knot_x, dim=-1, index=lower).squeeze(-1)
        spline_parameters["right"] = torch.gather(knot_x, dim=-1, index=upper).squeeze(-1)
        spline_parameters["bottom"] = torch.gather(knot_y, dim=-1, index=lower).squeeze(-1)
        spline_parameters["top"] = torch.gather(knot_y, dim=-1, index=upper).squeeze(-1)

        # gather all other parameter edges
        for key, value in parameters.items():
            if key in ["left", "bottom", "widths", "heights", "total_width"]:
                continue

            v = value[inside]

            spline_parameters[f"{key}_left"] = torch.gather(v, dim=-1, index=lower).squeeze(-1)
            spline_parameters[f"{key}_right"] = torch.gather(v, dim=-1, index=upper).squeeze(-1)

        if not rev:
            y = torch.clone(x)
            log_jac = y.new_zeros(y.shape)

            y[inside], log_jac[inside] = spline(x_in, spline_parameters, rev=rev)
            y[~inside], log_jac[~inside] = scale * x_out + shift, torch.log(scale)

            log_jac_det = utils.sum_except_batch(log_jac)

            y = y.movedim(-1, 1)

            return y, log_jac_det
        else:
            y = x
            y_in = x_in
            y_out = x_out

            x = torch.clone(y)
            log_jac = x.new_zeros(x.shape)

            x[inside], log_jac[inside] = spline(y_in, spline_parameters, rev=rev)
            x[~inside], log_jac[~inside] = (y_out - shift) / scale, torch.log(scale)

            log_jac_det = -utils.sum_except_batch(log_jac)

            x = x.movedim(-1, 1)

            return x, log_jac_det
