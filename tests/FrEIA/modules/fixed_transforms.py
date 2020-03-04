import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PermuteRandom(nn.Module):
    '''permutes input vector in a random but fixed way'''

    def __init__(self, dims_in, seed):
        super().__init__()

        self.in_channels = dims_in[0][0]

        np.random.seed(seed)
        self.perm = np.random.permutation(self.in_channels)
        np.random.seed()

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = torch.LongTensor(self.perm)
        self.perm_inv = torch.LongTensor(self.perm_inv)

    def forward(self, x, rev=False):
        if not rev:
            return [x[0][:, self.perm]]
        else:
            return [x[0][:, self.perm_inv]]

    def jacobian(self, x, rev=False):
        # TODO: use batch size, set as nn.Parameter so cuda() works
        return 0.

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class FixedLinearTransform(nn.Module):
    '''Fixed transformation according to y = Mx + b, with invertible
    matrix M.'''

    def __init__(self, dims_in, M, b):
        super().__init__()

        self.M = nn.Parameter(M.t(), requires_grad=False)
        self.M_inv = nn.Parameter(M.t().inverse(), requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

        self.logDetM = nn.Parameter(torch.log(torch.potrf(M).diag()).sum(),
                                    requires_grad=False)

    def forward(self, x, rev=False):
        if not rev:
            return [x[0].mm(self.M) + self.b]
        else:
            return [(x[0]-self.b).mm(self.M_inv)]

    def jacobian(self, x, rev=False):
        if rev:
            return -self.logDetM.expand(x[0].shape[0])
        else:
            return self.logDetM.expand(x[0].shape[0])

    def output_dims(self, input_dims):
        return input_dims

class Fixed1x1Conv(nn.Module):
    '''Fixed 1x1 conv transformation with matrix M.'''

    def __init__(self, dims_in, M):
        super().__init__()

        self.M = nn.Parameter(M.t().view(*M.shape, 1, 1), requires_grad=False)
        self.M_inv = nn.Parameter(M.t().inverse().view(*M.shape, 1, 1), requires_grad=False)

        self.logDetM = nn.Parameter(torch.log(torch.det(M).abs()).sum(),
                                    requires_grad=False)

    def forward(self, x, rev=False):
        if not rev:
            return [F.conv2d(x[0], self.M)]
        else:
            return [F.conv2d(x[0], self.M_inv)]

    def jacobian(self, x, rev=False):
        if rev:
            return -self.logDetM.expand(x[0].shape[0])
        else:
            return self.logDetM.expand(x[0].shape[0])

    def output_dims(self, input_dims):
        return input_dims

import warnings

def _deprecated_by(orig_class):
    class deprecated_class(orig_class):
        def __init__(self, *args, **kwargs):

            warnings.warn(F"{self.__class__.__name__} is deprecated and will be removed in the public release. "
                          F"Use {orig_class.__name__} instead.",
                          DeprecationWarning)
            super().__init__(*args, **kwargs)

    return deprecated_class

permute_layer = _deprecated_by(PermuteRandom)
linear_transform = _deprecated_by(FixedLinearTransform)
conv_1x1 = _deprecated_by(Fixed1x1Conv)
