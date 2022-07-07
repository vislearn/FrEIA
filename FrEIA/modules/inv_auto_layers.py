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

        x < 0 &\\implies g(x) = x \\odot \\exp(\\alpha_-)
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
    '''Scale each element of the input by a learned, non-negative factor.
    Unlike most other FrEIA modules, the scaling is not e.g. channel-wise for images,
    but really scales each individual element.
    To ensure positivity, the scaling is learned in log-space:

    .. math::

        g(x) = x \\odot \\exp(s)
    '''

    def __init__(self, dims_in, dims_c=None, init_scale=1.0):
        '''
        Args:
          init_scale: The initial scaling value. It accounts for the exp-activation, 
            i.e. :math:`\\exp(s) =` ``init_scale``.
        '''
        super().__init__(dims_in, dims_c)
        self.s = nn.Parameter(np.log(init_scale) * torch.ones(1, *dims_in[0]))

    def forward(self, x, rev=False, jac=True):

        if rev:
            scale = -self.s
        else:
            scale = self.s

        if jac:
            jac = torch.sum(self.s).unsqueeze(0)
        else:
            jac = None

        return [x[0] * torch.exp(scale)], jac

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims


class InvAutoFC(InvertibleModule):
    '''Fully connected 'Invertible Autoencoder'-layer (see arxiv.org/pdf/1802.06869.pdf).
    The weight matrix of the inverse is the tranposed weight matrix of the forward pass.
    If a reconstruction loss between forward and inverse is used, the layer converges
    to an invertible, orthogonal, linear transformation.
    '''

    def __init__(self, dims_in, dims_c=None, dims_out=None):
        '''
        Args:
          dims_out: If None, the output dimenison equals the input dimenison.
            However, becuase InvAuto is only asymptotically invertible, there is
            no strict limitation to have the same number of input- and
            ouput-dimensions. If dims_out is an integer instead of None,
            that number of output dimensions is used.
        '''
        super().__init__(dims_in, dims_c)
        self.dims_in = dims_in
        if dims_out is None:
            self.dims_out = dims_in[0][0]
        else:
            self.dims_out = dims_out

        self.weights = nn.Parameter(np.sqrt(1. / self.dims_out) * torch.randn(self.dims_out, self.dims_in[0][0]))
        self.bias = nn.Parameter(torch.randn(1, self.dims_out))
        print(self.weights.shape)
        print(self.bias.shape)

    def forward(self, x, rev=False, jac=True):
        if jac:
            warnings.warn('Invertible Autoencoder layers do not have a tractable log-det-Jacobian. '
                          'It approaches 0 at convergence, but the value may be incorrect duing training.')

        if not rev:
            return [f.linear(x[0], self.weights) + self.bias], 0.
        else:
            return [f.linear(x[0] - self.bias, self.weights.t())], 0.

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        if len(input_dims[0]) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use flattened (1D) input")
        return [(self.dims_out,)]


class InvAutoConv2D(InvertibleModule):
    '''Convolutional variant of the 'Invertible Autoencoder'-layer
    (see arxiv.org/pdf/1802.06869.pdf). The the inverse is a tranposed
    convolution with the same kernel as the forward pass. If a reconstruction
    loss between forward and inverse is used, the layer converges to an
    invertible, orthogonal, linear transformation.
    '''

    def __init__(self, dims_in, dims_c=None, dims_out=None, kernel_size=3, padding=1):
        '''
        Args:
          kernel_size: Spatial size of the convlution kernel.
          padding: Padding of the input. Choosing ``padding = kernel_size // 2`` retains
            the image shape between in- and output.
          dims_out: If None, the output dimenison equals the input dimenison.
            However, becuase InvAuto is only asymptotically invertible, there is
            no strict limitation to have the same number of input- and
            ouput-dimensions. Therefore dims_out can also be a tuple of length 3:
            (channels, width, height). The channels are the output channels of the
            convolution. The user is responsible for making the width and height match
            with the actual output, depending on kernel_size and padding.
        '''

        super().__init__(dims_in, dims_c)
        self.dims_in = dims_in

        if dims_out is None:
            self.dims_out = dims_in[0]
        else:
            self.dims_out = dims_out

        self.kernel_size = kernel_size
        self.padding = padding

        self.conv2d = nn.Conv2d(dims_in[0][0], self.dims_out[0], kernel_size=kernel_size, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.randn(1, self.dims_out[0], 1, 1))

    def forward(self, x, rev=False, jac=True):
        if jac:
            warnings.warn('Invertible Autoencoder layers do not have a tractable log-det-Jacobian.'
                          'It approaches 0 at convergence, but the value may be incorrect duing training.')

        if not rev:
            out = self.conv2d(x[0])
            out += self.bias
        else:
            out = x[0] - self.bias
            out = f.conv_transpose2d(out, self.conv2d.weight, bias=None, padding=self.padding)

        return [out], 0.

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError(f"{self.__class__.__name__} can only use image input (3D tensors)")
        return [self.dims_out]
