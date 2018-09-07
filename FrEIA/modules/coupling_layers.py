from math import exp

import torch
import torch.nn as nn

from .coeff_functs import F_conv, F_fully_connected


class rev_layer(nn.Module):
    '''General reversible layer modeled after the lifting scheme. Uses some
    non-reversible transformation F, but splits the channels up to make it
    revesible (see lifting scheme). F itself does not have to be revesible. See
    F_* classes above for examples.'''

    def __init__(self, dims_in, F_class=F_conv, F_args={}):
        super(rev_layer, self).__init__()
        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.F = F_class(self.split_len2, self.split_len1, **F_args)
        self.G = F_class(self.split_len1, self.split_len2, **F_args)

    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            y2 = x2 + self.G(y1)
        else:
            y2 = x2 - self.G(x1)
            y1 = x1 - self.F(y2)

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, rev=False):
        return torch.zeros(x.shape[0])

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class rev_multiplicative_layer(nn.Module):
    '''The RevNet block is not a general function approximator. The reversible
    layer with a multiplicative term presented in the real-NVP paper is much
    more general. This class uses some non-reversible transformation F, but
    splits the channels up to make it revesible (see lifting scheme). F itself
    does not have to be revesible. See F_* classes above for examples.'''

    def __init__(self, dims_in, F_class=F_fully_connected, F_args={},
                 clamp=5.):
        super(rev_multiplicative_layer, self).__init__()
        channels = dims_in[0][0]

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.ndims = len(dims_in[0])

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(self.split_len1, self.split_len2, **F_args)
        self.t1 = F_class(self.split_len1, self.split_len2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1, **F_args)
        self.t2 = F_class(self.split_len2, self.split_len1, **F_args)

    def e(self, s):
        # return torch.exp(torch.clamp(s, -self.clamp, self.clamp))
        # return (self.max_s-self.min_s) * torch.sigmoid(s) + self.min_s
        return torch.exp(self.clamp * 0.636 * torch.atan(s))

    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s)

    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = self.e(self.s2(x2)) * x1 + self.t2(x2)
            y2 = self.e(self.s1(y1)) * x2 + self.t1(y1)
        else:  # names of x and y are swapped!
            y2 = (x2 - self.t1(x1)) / self.e(self.s1(x1))
            y1 = (x1 - self.t2(y2)) / self.e(self.s2(y2))
        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            s2 = self.s2(x2)
            y1 = self.e(s2) * x1 + self.t2(x2)
            jac = self.log_e(self.s1(y1)) + self.log_e(s2)
        else:
            s1 = self.s1(x1)
            y2 = (x2 - self.t1(x1)) / self.e(s1)
            jac = -self.log_e(s1) - self.log_e(self.s2(y2))

        return torch.sum(jac, dim=tuple(range(1, self.ndims+1)))

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class glow_coupling_layer(nn.Module):
    def __init__(self, dims_in, F_class=F_fully_connected, F_args={},
                 clamp=5.):
        super(glow_coupling_layer, self).__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(self.split_len1, self.split_len2*2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1*2, **F_args)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r2 = self.s2(x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1

        else:  # names of x and y are swapped!
            r1 = self.s1(x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r2 = self.s2(x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]

        else:  # names of x and y are swapped!
            r1 = self.s1(x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]

        jac = (torch.sum(self.log_e(s1), dim=1)
               + torch.sum(self.log_e(s2), dim=1))
        for i in range(self.ndims-1):
            jac = torch.sum(jac, dim=1)

        return jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims
