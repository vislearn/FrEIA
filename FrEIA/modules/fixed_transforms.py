from . import InvertibleModule

from typing import Union, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PermuteRandom(InvertibleModule):
    '''Constructs a random permutation, that stays fixed during training.
    Permutes along the first (channel-) dimension for multi-dimenional tensors.'''

    def __init__(self, dims_in, dims_c=None, seed: Union[int, None] = None):
        '''Additional args in docstring of base class FrEIA.modules.InvertibleModule.

        Args:
          seed: Int seed for the permutation (numpy is used for RNG). If seed is
            None, do not reseed RNG.
        '''
        super().__init__(dims_in, dims_c)

        self.in_channels = dims_in[0][0]

        if seed is not None:
            np.random.seed(seed)
        self.perm = np.random.permutation(self.in_channels)

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = nn.Parameter(torch.LongTensor(self.perm), requires_grad=False)
        self.perm_inv = nn.Parameter(torch.LongTensor(self.perm_inv), requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        if not rev:
            return [x[0][:, self.perm]], 0.
        else:
            return [x[0][:, self.perm_inv]], 0.

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims


class FixedLinearTransform(InvertibleModule):
    '''Fixed linear transformation for 1D input tesors. The transformation is
    :math:`y = Mx + b`. With *d* input dimensions, *M* must be an invertible *d x d* tensor,
    and *b* is an optional offset vector of length *d*.'''

    def __init__(self, dims_in, dims_c=None, M: torch.Tensor = None,
                 b: Union[None, torch.Tensor] = None):
        '''Additional args in docstring of base class FrEIA.modules.InvertibleModule.

        Args:
          M: Square, invertible matrix, with which each input is multiplied. Shape ``(d, d)``.
          b: Optional vector which is added element-wise. Shape ``(d,)``.
        '''
        super().__init__(dims_in, dims_c)

        # TODO: it should be possible to give conditioning instead of M, so that the condition
        # provides M and b on each forward pass.

        if M is None:
            raise ValueError("Need to specify the M argument, the matrix to be multiplied.")

        self.M = nn.Parameter(M.t(), requires_grad=False)
        self.M_inv = nn.Parameter(M.t().inverse(), requires_grad=False)

        if b is None:
            self.b = 0.
        else:
            self.b = nn.Parameter(b.unsqueeze(0), requires_grad=False)

        self.logDetM = nn.Parameter(torch.slogdet(M)[1], requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        j = self.logDetM.expand(x[0].shape[0])
        if not rev:
            out = x[0].mm(self.M) + self.b
            return (out,), j
        else:
            out = (x[0] - self.b).mm(self.M_inv)
            return (out,), -j

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims


class Fixed1x1Conv(InvertibleModule):
    '''Given an invertible matrix M, a 1x1 convolution is performed using M as
    the convolution kernel. Effectively, a matrix muplitplication along the
    channel dimension is performed in each pixel.'''

    def __init__(self, dims_in, dims_c=None, M: torch.Tensor = None):
        '''Additional args in docstring of base class FrEIA.modules.InvertibleModule.

        Args:
          M: Square, invertible matrix, with which each input is multiplied. Shape ``(d, d)``.
        '''
        super().__init__(dims_in, dims_c)

        # TODO: it should be possible to give conditioning instead of M, so that the condition
        # provides M and b on each forward pass.

        if M is None:
            raise ValueError("Need to specify the M argument, the matrix to be multiplied.")

        self.M = nn.Parameter(M.t().view(*M.shape, 1, 1), requires_grad=False)
        self.M_inv = nn.Parameter(M.t().inverse().view(*M.shape, 1, 1), requires_grad=False)
        self.logDetM = nn.Parameter(torch.slogdet(M)[1], requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        n_pixels = x[0][0, 0].numel()
        j = self.logDetM * n_pixels

        if not rev:
            return (F.conv2d(x[0], self.M),), j
        else:
            return (F.conv2d(x[0], self.M_inv),), -j

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError(f"{self.__class__.__name__} requires 3D input (channels, height, width)")
        return input_dims


class InvertibleSigmoid(InvertibleModule):
    '''Applies the sigmoid function element-wise across all batches, and the associated
    inverse function in reverse pass. Contains no trainable parameters.
    Sigmoid function S(x) and its corresponding inverse function is given by

    .. math::

        S(x)      &= \\frac{1}{1 + \\exp(-x)} \\\\
        S^{-1}(x) &= \\log{\\frac{x}{1-x}}.

    The returning Jacobian is computed as

    .. math::

        J = \\log \\det \\frac{1}{(1+\\exp{x})(1+\\exp{-x})}.

    '''
    def __init__(self, dims_in, **kwargs):
        super().__init__(dims_in, **kwargs)

    def output_dims(self, dims_in):
        return dims_in

    def forward(self, x_or_z: Iterable[torch.Tensor], c: Iterable[torch.Tensor] = None,
                rev: bool = False, jac: bool = True) \
            -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        x_or_z = x_or_z[0]
        if not rev:
            # S(x)
            result = 1 / (1 + torch.exp(-x_or_z))
        else:
            # S^-1(z)
            # only defined within range 0-1, non-inclusive; else, it will returns nan.
            result = torch.log(x_or_z / (1 - x_or_z))
        if not jac:
            return (result, )

        # always compute jacobian using the forward direction, but with different inputs
        _input = result if rev else x_or_z
        # the following is the diagonal Jacobian as sigmoid is an element-wise op
        logJ = torch.log(1 / ((1 + torch.exp(_input)) * (1 + torch.exp(-_input))))
        # determinant of a log diagonal Jacobian is simply the sum of its diagonals
        detLogJ = logJ.sum(1)
        if not rev:
            return ((result, ), detLogJ)
        else:
            return ((result, ), -detLogJ)
