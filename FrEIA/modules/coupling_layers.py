from math import exp

import torch
import torch.nn as nn

from .coeff_functs import F_conv, F_fully_connected


class NICECouplingBlock(nn.Module):
    '''Coupling Block following the NICE design.

    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.'''

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None):
        super().__init__()

        channels = dims_in[0][0]
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        assert all([dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.F = subnet_constructor(self.split_len2 + condition_length, self.split_len1)
        self.G = subnet_constructor(self.split_len1 + condition_length, self.split_len2)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1 = x1 + self.F(x2_c)
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2 = x2 + self.G(y1_c)
        else:
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2 = x2 - self.G(x1_c)
            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1 = x1 - self.F(y2_c)

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class RNVPCouplingBlock(nn.Module):
    '''Coupling Block following the RealNVP design.

    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.
    clamp:              Soft clamping for the multiplicative component. The amplification or attenuation
                        of each input dimension can be at most ±exp(clamp).'''


    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=5.):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        assert all([dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2)
        self.t1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1)
        self.t2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1)

    def e(self, s):
        # return torch.exp(torch.clamp(s, -self.clamp, self.clamp))
        # return (self.max_s-self.min_s) * torch.sigmoid(s) + self.min_s
        return torch.exp(self.clamp * 0.636 * torch.atan(s))

    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            s2, t2 = self.s2(x2_c), self.t2(x2_c)
            y1 = self.e(s2) * x1 + t2
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            s1, t1 = self.s1(y1_c), self.t1(y1_c)
            y2 = self.e(s1) * x2 + t1
            self.last_s = [s1, s2]
        else:  # names of x and y are swapped!
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            s1, t1 = self.s1(x1_c), self.t1(x1_c)
            y2 = (x2 - t1) / self.e(s1)
            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            s2, t2 = self.s2(y2_c), self.t2(y2_c)
            y1 = (x1 - t2) / self.e(s2)
            self.last_s = [s1, s2]

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            jac1 = torch.sum(self.log_e(self.last_s[0]), dim=tuple(range(1, self.ndims+1)))
            jac2 = torch.sum(self.log_e(self.last_s[1]), dim=tuple(range(1, self.ndims+1)))
        else:
            jac1 = -torch.sum(self.log_e(self.last_s[0]), dim=tuple(range(1, self.ndims+1)))
            jac2 = -torch.sum(self.log_e(self.last_s[1]), dim=tuple(range(1, self.ndims+1)))

        return jac1 + jac2

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class GLOWCouplingBlock(nn.Module):
    '''Coupling Block following the GLOW design. The only difference to the RealNVP coupling blocks,
    is the fact that it uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate
    subnetworks. This reduces computational cost and speeds up learning.

    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.
    clamp:              Soft clamping for the multiplicative component. The amplification or attenuation
                        of each input dimension can be at most ±exp(clamp).'''

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=5.):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            F"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2*2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1*2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r2 = self.s2(torch.cat([x2, *c], 1) if self.conditional else x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, *c], 1) if self.conditional else y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = (  torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims+1)))
                             + torch.sum(self.log_e(s2), dim=tuple(range(1, self.ndims+1))))

        else: # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, *c], 1) if self.conditional else x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(torch.cat([y2, *c], 1) if self.conditional else y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = (- torch.sum(self.log_e(s1), dim=tuple(range(1, self.ndims+1)))
                             - torch.sum(self.log_e(s2), dim=tuple(range(1, self.ndims+1))))

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims


class GINCouplingBlock(nn.Module):
    '''Coupling Block following the GIN design. The difference from the RealNVP coupling blocks
    is that it uses a single subnetwork (like the GLOW coupling blocks) to jointly predict [s_i, t_i], 
    instead of two separate subnetworks, and the Jacobian determinant is constrained to be 1. 
    This constrains the block to be volume-preserving. Volume preservation is achieved by subtracting
    the mean of the output of the s subnetwork from itself. 
    Note: this implementation differs slightly from the originally published implementation, which 
    scales the final component of the s subnetwork so the sum of the outputs of s is zero. There was
    no difference found between the implementations in practice, but subtracting the mean guarantees 
    that all outputs of s are at most ±exp(clamp), which might be more stable in certain cases.

    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.
    clamp:              Soft clamping for the multiplicative component. The amplification or attenuation
                        of each input dimension can be at most ±exp(clamp).'''

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=5.):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            F"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2*2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1*2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r2 = self.s2(torch.cat([x2, *c], 1) if self.conditional else x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            s2 = self.log_e(s2)
            s2 -= s2.mean(1, keepdim=True)
            y1 = torch.exp(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, *c], 1) if self.conditional else y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            s1 = self.log_e(s1)
            s1 -= s1.mean(1, keepdim=True)
            y2 = torch.exp(s1) * x2 + t1
            
            self.last_jac = (  torch.sum(s1, dim=tuple(range(1, self.ndims+1)))
                             + torch.sum(s2, dim=tuple(range(1, self.ndims+1))))

        else: # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, *c], 1) if self.conditional else x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            s1 = self.log_e(s1)
            s1 -= s1.mean(1, keepdim=True)
            y2 = (x2 - t1) * torch.exp(-s1)

            r2 = self.s2(torch.cat([y2, *c], 1) if self.conditional else y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            s2 = self.log_e(s2)
            s2 -= s2.mean(1, keepdim=True)
            y1 = (x1 - t2) * torch.exp(-s2)
            self.last_jac = (- torch.sum(s1, dim=tuple(range(1, self.ndims+1)))
                             - torch.sum(s2, dim=tuple(range(1, self.ndims+1))))

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims


class AffineCouplingOneSided(nn.Module):
    '''Half of a coupling block following the RealNVP design (only one affine transformation on half
    the inputs). If random permutations or orthogonal transforms are used after every block, this is
    not a restriction and simplifies the design.

    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.
    clamp:              Soft clamping for the multiplicative component. The amplification or attenuation
                        of each input dimension can be at most ±exp(clamp).'''

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=5.):
        super().__init__()

        self.channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_idx = self.channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        assert all([dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s = subnet_constructor(self.split_idx + condition_length, self.channels - self.split_idx)
        self.t = subnet_constructor(self.split_idx + condition_length, self.channels - self.split_idx)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s))

    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s)

    def forward(self, x, c=[], rev=False):
        x1, x2 = torch.split(x[0], [self.split_idx, self.channels - self.split_idx], dim=1)

        if not rev:
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            s, t = self.s(x1_c), self.t(x1_c)
            y2 = self.e(s) * x2 + t
            self.last_s = s
            return [torch.cat((x1, y2), 1)]
        else:
            y1, y2 = x1, x2
            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            s, t = self.s(y1_c), self.t(y1_c)
            x2 = (y2 - t) / self.e(s)
            self.last_s = s
            return [torch.cat((y1, x2), 1)]

    def jacobian(self, x, c=[], rev=False):
        x1, x2 = torch.split(x[0], [self.split_idx, self.channels - self.split_idx], dim=1)

        if not rev:
            jac = self.log_e(self.last_s)
        else:
            jac = -self.log_e(self.last_s)

        return torch.sum(jac, dim=tuple(range(1, self.ndims+1)))

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use one input."
        return input_dims


class ConditionalAffineTransform(nn.Module):
    '''Similar to SPADE: Perform an affine transformation on the whole input,
    determined through the condition

    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.
    clamp:              Soft clamping for the multiplicative component. The amplification or attenuation
                        of each input dimension can be at most ±exp(clamp).'''

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=5.):
        super().__init__()

        self.ndims = len(dims_in[0])
        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s = subnet_constructor(dims_c[0][0], dims_in[0][0])
        self.t = subnet_constructor(dims_c[0][0], dims_in[0][0])

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s))

    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s)

    def forward(self, x, c=[], rev=False):
        s, t = self.s(c[0]), self.t(c[0])
        self.last_s = s
        if not rev:
            return [self.e(s) * x[0] + t]
        else:
            return [(x[0] - t) / self.e(s)]

    def jacobian(self, x, c=[], rev=False):
        if not rev:
            jac = self.log_e(self.last_s)
        else:
            jac = -self.log_e(self.last_s)

        return torch.sum(jac, dim=tuple(range(1, self.ndims+1)))

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use exactly two inputs."
        return [input_dims[0]]

import warnings
def _deprecated_by(orig_class):
    class deprecated_class(orig_class):
        def __init__(self, dims_in, dims_c=[], F_class=F_fully_connected, F_args={}, **kwargs):
            warnings.warn(F"{self.__class__.__name__} is deprecated and will be removed in the public release. "
                          F"Use {orig_class.__name__} instead.",
                          DeprecationWarning)

            def coeff_func_wrapper(ch_in, ch_out):
                return F_class(ch_in, ch_out, **F_args)

            super().__init__(dims_in, dims_c, subnet_constructor=coeff_func_wrapper, **kwargs)

    return deprecated_class

rev_layer = _deprecated_by(NICECouplingBlock)
rev_multiplicative_layer = _deprecated_by(RNVPCouplingBlock)
glow_coupling_layer = _deprecated_by(GLOWCouplingBlock)
AffineCoupling = _deprecated_by(AffineCouplingOneSided)
ExternalAffineCoupling = _deprecated_by(ConditionalAffineTransform)
