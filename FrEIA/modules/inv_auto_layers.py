from . import InvertibleModule

import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class InvAutoActTwoSided(InvertibleModule):
    '''A nonlinear invertible activation analogous to Leaky ReLU, with
    learned slopes.

    The slopes are learned separately for each entry along the first
    intput dimenison (after the batch dimenison). I.e. element-wise for
    flattened inputs, channel-wise for image inputs, etc.
    Internally, the slopes are learned in log-space, to ensure they stay
    strictly > 0:

    .. math::

        x \\geq 0 &\\implies g(x) = x \\odot \\exp(\\alpha_+)

        x < 0 &\\implies g(x) = x \\odot \\exp(\\alpha_-) x
    '''

    def __init__(self, dims_in, dims_c=None, init_pos: float = 2.0, init_neg: float = 0.5, learnable: bool = True):
        '''
        Args:
          init_pos: The initial slope for the positive half of the activation. Must be > 0.
            Note that the initial value accounts for the exp-activation, meaning
            :math:`\\exp(\\alpha_+) =` ``init_pos``.
          init_pos: The initial slope for the negative half of the activation. Must be > 0.
            The initial value accounts for the exp-activation the same as init_pos.
          learnable: If False, the slopes are fixed at their initial value, and not learned.
        '''
        super().__init__(dims_in, dims_c)
        self.tensor_rank = len(dims_in[0])

        self.alpha_pos = np.log(init_pos) * torch.ones(dims_in[0][0])
        self.alpha_pos = self.alpha_pos.view(1, -1, *([1] * (self.tensor_rank - 1)))
        self.alpha_pos = nn.Parameter(self.alpha_pos)

        self.alpha_neg = np.log(init_neg) * torch.ones(dims_in[0][0])
        self.alpha_neg = self.alpha_neg.view(1, -1, *([1] * (self.tensor_rank - 1)))
        self.alpha_neg = nn.Parameter(self.alpha_neg)

        if not learnable:
            self.alpha_pos.requires_grad = False
            self.alpha_neg.requires_grad = False

    def forward(self, x, rev=False, jac=True):

        log_slope = self.alpha_pos + 0.5 * (self.alpha_neg - self.alpha_pos) * (1 - x[0].sign())
        if rev:
            log_slope *= -1

        if jac:
            j = torch.sum(log_slope, dim=tuple(range(1, self.tensor_rank + 1)))
        else:
            j = None

        return [x[0] * torch.exp(log_slope)], j

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims


class InvAutoAct(InvertibleModule):
    '''A nonlinear invertible activation analogous to Leaky ReLU, with
    learned slopes.

    The slope is symmetric between the positive and negative side, i.e.

    .. math::

        x \\geq 0 &\\implies g(x) = x \\odot \\exp(\\alpha)

        x < 0 &\\implies g(x) = x \\oslash \\exp(\\alpha)

    A separate slope is learned for each entry along the first
    intput dimenison (after the batch dimenison). I.e. element-wise for
    flattened inputs, channel-wise for image inputs, etc.
    '''

    def __init__(self, dims_in, dims_c=None, slope_init=2.0, learnable=True):
        '''
        Args:
          slope_init: The initial value of the slope on the positive side.
            Accounts for the exp-activation, i.e. :math:`\\exp(\\alpha) =` ``slope_init``.
          learnable: If False, the slopes are fixed at their initial value, and not learned.
        '''
        super().__init__(dims_in, dims_c)

        self.tensor_rank = len(dims_in[0])
        self.alpha = np.log(slope_init) * torch.ones(1, dims_in[0][0], *([1] * (len(dims_in[0]) - 1)))
        self.alpha = nn.Parameter(self.alpha)

        if not learnable:
            self.alpha.requires_grad = False

    def forward(self, x, rev=False, jac=True):
        log_slope = self.alpha * x[0].sign()
        if rev:
            log_slope *= -1

        if jac:
            j = torch.sum(log_slope, dim=tuple(range(1, self.tensor_rank + 1)))
        else:
            j = None

        return [x[0] * torch.exp(log_slope)], j

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims


class InvAutoActFixed(InvAutoAct):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn("Deprecated: please use InvAutoAct with the learnable=False argument.")


class LearnedElementwiseScaling(InvertibleModule):

    def __init__(self, dims_in, dims_c=None):
        super().__init__(dims_in, dims_c)
        self.s = nn.Parameter(torch.zeros(*dims_in[0]))

    def forward(self, x, rev=False, jac=True):
        if jac:
            jac = torch.sum(self.s).unsqueeze(0)
            if rev:
                jac *= -1
        else:
            jac = None

        if not rev:
            return [x[0] * self.s.exp()], jac
        else:
            return [x[0] * self.s.neg().exp_()], jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class InvAutoFC(InvertibleModule):

    def __init__(self, dims_in, dims_c, dims_out=None):
        super().__init__(dims_in, dims_c)
        self.dims_in = dims_in
        if dims_out is None:
            self.dims_out = deepcopy(dims_in)
        else:
            self.dims_out = dims_out

        self.weights = nn.Parameter(0.01 * torch.randn(self.dims_out[0][0], self.dims_in[0][0]))
        self.bias = nn.Parameter(0.01 * torch.randn(1, self.dims_out[0][0]))

    def forward(self, x, rev=False, jac=True):
        if jac:
            raise NotImplementedError("TODO: Jacobian is not implemented for InvAutoFC")

        if not rev:
            return [f.linear(x[0], self.weights) + self.bias.expand(x[0].size()[0], *self.dims_out[0])], None
        else:
            return [f.linear(x[0] - self.bias.expand(x[0].size()[0], *self.dims_out[0]), self.weights.t())], None

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return self.dims_out


class InvAutoConv2D(InvertibleModule):

    def __init__(self, dims_in, dims_c=None, dims_out=None, kernel_size=3, padding=1):
        super().__init__(dims_in, dims_c)
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv2d = nn.Conv2d(dims_in[0][0], dims_out[0][0], kernel_size=kernel_size, padding=padding, bias=False)
        self.bias = nn.Parameter(0.01 * torch.randn(1, dims_out[0][0], 1, 1))

    def forward(self, x, rev=False, jac=True):
        if jac:
            raise NotImplementedError("TODO: Jacobian is not implemented for InvAutoConv2D")

        if not rev:
            out = self.conv2d(x[0])
            out += self.bias.expand(out.size())
        else:
            out = x[0] - self.bias.expand(x[0].size())
            out = f.conv_transpose2d(out, self.conv2d.weight, bias=None, padding=self.padding)

        return [out], None

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return self.dims_out
