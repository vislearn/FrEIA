from . import InvertibleModule

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PermuteRandom(InvertibleModule):
    '''permutes input vector in a random but fixed way'''

    def __init__(self, dims_in, dims_c=None, seed=None):
        super().__init__(dims_in, dims_c)

        self.in_channels = dims_in[0][0]

        if seed is not None:
            np.random.seed(seed)
        self.perm = np.random.permutation(self.in_channels)

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = torch.LongTensor(self.perm)
        self.perm_inv = torch.LongTensor(self.perm_inv)

    def forward(self, x, rev=False, jac=True):
        if not rev:
            return [x[0][:, self.perm]], 0.
        else:
            return [x[0][:, self.perm_inv]], 0.

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class FixedLinearTransform(InvertibleModule):
    '''Fixed transformation according to y = Mx + b, with invertible
    matrix M.'''

    def __init__(self, dims_in, dims_c=None, M=None, b=None):
        super().__init__(dims_in, dims_c)

        self.M = nn.Parameter(M.t(), requires_grad=False)
        self.M_inv = nn.Parameter(M.t().inverse(), requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

        self.logDetM = nn.Parameter(torch.slogdet(M)[1], requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        if not rev:
            return [x[0].mm(self.M) + self.b], self.logDetM.expand(x[0].shape[0])
        else:
            return [(x[0]-self.b).mm(self.M_inv)], -self.logDetM.expand(x[0].shape[0])

    def output_dims(self, input_dims):
        return input_dims

class Fixed1x1Conv(InvertibleModule):
    '''Fixed 1x1 conv transformation with matrix M.'''

    def __init__(self, dims_in, dims_c=None, M=None):
        super().__init__(dims_in, dims_c)

        self.M = nn.Parameter(M.t().view(*M.shape, 1, 1), requires_grad=False)
        self.M_inv = nn.Parameter(M.t().inverse().view(*M.shape, 1, 1), requires_grad=False)

        self.logDetM = nn.Parameter(torch.log(torch.det(M).abs()).sum(),
                                    requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        # TODO: is the jacobian wrong??
        if not rev:
            return [F.conv2d(x[0], self.M)], self.logDetM.expand(x[0].shape[0])
        else:
            return [F.conv2d(x[0], self.M_inv)], -self.logDetM.expand(x[0].shape[0])

    def output_dims(self, input_dims):
        return input_dims
