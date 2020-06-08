from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f


class InvAutoActTwoSided(nn.Module):

    def __init__(self, dims_in, clamp=5.):
        super().__init__()
        self.clamp = clamp
        self.alpha_pos = nn.Parameter(0.05 * torch.randn(dims_in[0][0]) + 0.7)
        self.alpha_neg = nn.Parameter(0.05 * torch.randn(dims_in[0][0]) - 0.7)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s/self.clamp))

    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s/self.clamp)

    def forward(self, x, rev=False):
        if not rev:
            return [x[0] * self.e(self.alpha_pos + 0.5 * (self.alpha_neg - self.alpha_pos) * (1 - x[0].sign()))]
        else:
            return [x[0] * self.e(-self.alpha_pos - 0.5 * (self.alpha_neg - self.alpha_pos) * (1 - x[0].sign()))]

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims

    def jacobian(self, x, rev=False):
       return (-1)**rev * torch.sum(self.log_e(self.alpha_pos + 0.5 * (self.alpha_neg - self.alpha_pos) * (1 - x[0].sign())), dim=1)

class InvAutoAct(nn.Module):

    def __init__(self, dims_in):
        super().__init__()
        self.alpha = nn.Parameter(0.01 * torch.randn(dims_in[0][0]) + 0.7)

    def forward(self, x, rev=False):
        if not rev:
            return [x[0] * torch.exp(self.alpha * x[0].sign())]
        else:
            return [x[0] * torch.exp(self.alpha * x[0].sign().neg_())]

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class InvAutoActFixed(nn.Module):

    def __init__(self, dims_in, alpha=2.0):
        super().__init__()
        self.alpha = alpha
        self.alpha_inv = 1. / alpha

        self.log_alpha = np.log(alpha)

    def forward(self, x, rev=False):
        if not rev:
            return [self.alpha_inv * f.leaky_relu(x[0], self.alpha*self.alpha)]
        else:
            return [self.alpha * f.leaky_relu(x[0], self.alpha_inv*self.alpha_inv)]

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims

    def jacobian(self, x, rev=False):
        return (-1)**rev * torch.sum(self.log_alpha * x[0].sign(), dim=1)


class LearnedElementwiseScaling(nn.Module):

    def __init__(self, dims_in):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(*dims_in[0]))

    def forward(self, x, rev=False):
        if not rev:
            return [x[0] * self.s.exp()]
        else:
            return [x[0] * self.s.neg().exp_()]

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class InvAutoFC(nn.Module):

    def __init__(self, dims_in, dims_out=None):
        super().__init__()
        self.dims_in = dims_in
        if dims_out is None:
            self.dims_out = deepcopy(dims_in)
        else:
            self.dims_out = dims_out

        self.weights = nn.Parameter(0.01 * torch.randn(self.dims_out[0][0], self.dims_in[0][0]))
        self.bias = nn.Parameter(0.01 * torch.randn(1, self.dims_out[0][0]))

    def forward(self, x, rev=False):
        if not rev:
            return [f.linear(x[0], self.weights) + self.bias.expand(x[0].size()[0], *self.dims_out[0])]
        else:
            return [f.linear(x[0] - self.bias.expand(x[0].size()[0], *self.dims_out[0]), self.weights.t())]

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return self.dims_out


class InvAutoConv2D(nn.Module):

    def __init__(self, dims_in, dims_out, kernel_size=3, padding=1):
        super().__init__()
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv2d = nn.Conv2d(dims_in[0][0], dims_out[0][0], kernel_size=kernel_size, padding=padding, bias=False)
        self.bias = nn.Parameter(0.01 * torch.randn(1, dims_out[0][0], 1, 1))

    def forward(self, x, rev=False):
        if not rev:
            out = self.conv2d(x[0])
            out += self.bias.expand(out.size())
        else:
            out = x[0] - self.bias.expand(x[0].size())
            out = f.conv_transpose2d(out, self.conv2d.weight, bias=None, padding=self.padding)

        return [out]

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return self.dims_out
