import pdb
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group

class AllInOneBlock(nn.Module):
    ''' Combines affine coupling, permutation, global affine transformation ('ActNorm')
    in one block.'''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor=None,
                 affine_clamping=2.,
                 gin_block=False,
                 global_affine_init=1.,
                 global_affine_type='SOFTPLUS',
                 permute_soft=False,
                 learned_householder_permutation=0,
                 reverse_permutation=False):
        '''
        subnet_constructor: class or callable f, called as
            f(channels_in, channels_out) and should return a torch.nn.Module

        affine_clamping: clamp the output of the mutliplicative coefficients
            (before exponentiation) to +/- affine_clamping.

        gin_block: Turn the block into a GIN block from Sorrenson et al, 2019

        global_affine_init: Initial value for the global affine scaling beta

        global_affine_init: 'SIGMOID', 'SOFTPLUS', or 'EXP'. Defines the activation
            to be used on the beta for the global affine scaling.

        permute_soft: bool, whether to sample the permutation matrices from SO(N),
            or to use hard permutations in stead. Note, permute_soft=True is very slow
            when working with >512 dimensions.

        learned_householder_permutation: Int, if >0,  use that many learned householder
            reflections. Slow if large number. Dubious whether it actually helps.

        reverse_permutation: Reverse the permutation before the block, as introduced by
            Putzky et al, 2019.
        '''

        super().__init__()

        channels = dims_in[0][0]
        self.Ddim = len(dims_in[0]) - 1
        self.sum_dims = tuple(range(1, 2 + self.Ddim))

        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        elif len(dims_c) == 1:
            self.conditional = True
            self.condition_channels = dims_c[0][0]
            assert tuple(dims_c[0][1:]) == tuple(dims_in[0][1:]), \
                F"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
        else:
            raise ValueError('Only supports one condition (concatenate externally)')

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]


        try:
            self.permute_function = {0 : F.linear,
                                     1 : F.conv1d,
                                     2 : F.conv2d,
                                     3 : F.conv3d}[self.Ddim]
        except KeyError:
            raise ValueError(f"Data has {1 + self.Ddim} dimensions. Must be 1-4.")

        self.in_channels = channels
        self.clamp = affine_clamping
        self.GIN = gin_block
        self.welling_perm = reverse_permutation
        self.householder = learned_householder_permutation

        if permute_soft and channels > 512:
            warnings.warn(("Soft permutation will take a very long time to initialize "
                           f"with {channels} feature channels. Consider using hard permutation instead."))

        if global_affine_type == 'SIGMOID':
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = (lambda a: 10 * torch.sigmoid(a - 2.))
        elif global_affine_type == 'SOFTPLUS':
            global_scale = 10. * global_affine_init
            self.softplus = nn.Softplus(beta=0.5)
            self.global_scale_activation = (lambda a: 0.1 * self.softplus(a))
        elif global_affine_type == 'EXP':
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = (lambda a: torch.exp(a))
        else:
            raise ValueError('Please, SIGMOID, SOFTPLUS or EXP, as global affine type')

        self.global_scale = nn.Parameter(torch.ones(1, self.in_channels, *([1] * self.Ddim)) * float(global_scale))
        self.global_offset = nn.Parameter(torch.zeros(1, self.in_channels,  *([1] * self.Ddim)))

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels,channels))
            for i,j in enumerate(np.random.permutation(channels)):
                w[i,j] = 1.

        if self.householder:
            self.vk_householder = nn.Parameter(0.2 * torch.randn(self.householder, channels), requires_grad=True)
            self.w = None
            self.w_inv = None
            self.w_0 = nn.Parameter(torch.FloatTensor(w), requires_grad=False)
        else:
            self.w = nn.Parameter(torch.FloatTensor(w).view(channels, channels, *([1] * self.Ddim)),
                                  requires_grad=False)
            self.w_inv = nn.Parameter(torch.FloatTensor(w.T).view(channels, channels, *([1] * self.Ddim)),
                                  requires_grad=False)

        self.s = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None

    def construct_householder_permutation(self):
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(w, torch.eye(self.in_channels).to(w.device) - 2 * torch.ger(vk, vk) / torch.dot(vk, vk))

        for i in range(self.Ddim):
            w = w.unsqueeze(-1)
        return w

    def log_e(self, s):
        s = self.clamp * torch.tanh(0.1 * s)
        if self.GIN:
            s -= torch.mean(s, dim=self.sum_dims, keepdim=True)
        return s

    def permute(self, x, rev=False):
        if self.GIN:
            scale = 1.
        else:
            scale = self.global_scale_activation( self.global_scale)
        if rev:
            return (self.permute_function(x, self.w_inv) - self.global_offset) / scale
        else:
            return self.permute_function(x * scale + self.global_offset, self.w)

    def pre_permute(self, x, rev=False):
        if rev:
            return self.permute_function(x, self.w)
        else:
            return self.permute_function(x, self.w_inv)

    def affine(self, x, a, rev=False):
        ch = x.shape[1]
        sub_jac = self.log_e(a[:,:ch])
        if not rev:
            return (x * torch.exp(sub_jac) + 0.1 * a[:,ch:],
                    torch.sum(sub_jac, dim=self.sum_dims))
        else:
            return ((x - 0.1 * a[:,ch:]) * torch.exp(-sub_jac),
                    -torch.sum(sub_jac, dim=self.sum_dims))

    def forward(self, x, c=[], rev=False):
        if self.householder:
            self.w = self.construct_householder_permutation()
            if rev or self.welling_perm:
                self.w_inv = self.w.transpose(0,1).contiguous()

        if rev:
            x = [self.permute(x[0], rev=True)]
        elif self.welling_perm:
            x = [self.pre_permute(x[0], rev=False)]

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1

        if not rev:
            a1 = self.s(x1c)
            x2, j2 = self.affine(x2, a1)
        else:
            # names of x and y are swapped!
            a1 = self.s(x1c)
            x2, j2 = self.affine(x2, a1, rev=True)

        self.last_jac = j2
        x_out = torch.cat((x1, x2), 1)

        n_pixels = 1
        for d in self.sum_dims[1:]:
            n_pixels *= x_out.shape[d]

        self.last_jac += ((-1)**rev * n_pixels) * (1 - int(self.GIN)) * (torch.log(self.global_scale_activation(self.global_scale) + 1e-12).sum())

        if not rev:
            x_out = self.permute(x_out, rev=False)
        elif self.welling_perm:
            x_out = self.pre_permute(x_out, rev=True)

        return [x_out]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims
