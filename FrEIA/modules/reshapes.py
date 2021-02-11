from . import InvertibleModule

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IRevNetDownsampling(InvertibleModule):
    '''The invertible spatial downsampling used in i-RevNet.
    Each group of four neighboring pixels is reorderd into one pixel with four times
    the channels in a checkerboard-like pattern. See i-RevNet, Jacobsen 2018 et al.
    '''

    def __init__(self, dims_in, dims_c=None, legacy_backend: bool = False):
        '''See docstring of base class (FrEIA.modules.InvertibleModule) for more.
        Args:
          legacy_backend: If True, uses the splitting and concatenating method,
            adapted from
            github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py
            for the use in FrEIA. Is usually slower on GPU.
            If False, uses a 2d strided convolution with a kernel representing
            the downsampling.  Note that the ordering of the output channels
            will be different.  If pixels in each patch in channel 1 
            are a1, b1, ..., and in channel 2 are a2, b2, ...
            Then the output channels will be the following:
            legacy_backend=True: a1, a2, ..., b1, b2, ..., c1, c2, ...
            legacy_backend=False: a1, b1, ..., a2, b2, ..., a3, b3, ...
            (see also order_by_wavelet in module HaarDownsampling)
            Usually this difference is completely irrelevant.
        '''
        super().__init__(dims_in, dims_c)

        self.channels = dims_in[0][0]
        self.block_size = 2
        self.block_size_sq = self.block_size**2
        self.legacy_backend = legacy_backend

        if not self.legacy_backend:
            # this kernel represents the reshape:
            # it apllies to 2x2 patches (stride 2), and transforms each
            # input channel to 4 channels.
            # The input value is tranferred whereever the kernel is 1.
            # (hence the indexing pattern 00, 01, 10, 11 represents the
            # cherckerboard.
            # For the upsampling, a transposed convolution is used for the 
            # opposite effect.

            self.downsample_kernel = torch.zeros(4,1,2,2)

            self.downsample_kernel[0, 0, 0, 0] = 1
            self.downsample_kernel[1, 0, 0, 1] = 1
            self.downsample_kernel[2, 0, 1, 0] = 1
            self.downsample_kernel[3, 0, 1, 1] = 1

            self.downsample_kernel = torch.cat([self.downsample_kernel] * self.channels, 0)
            self.downsample_kernel = nn.Parameter(self.downsample_kernel)
            self.downsample_kernel.requires_grad = False

    def forward(self, x, c=None, rev=False, jac=True):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        input = x[0]
        if not rev:
            if self.legacy_backend:
                # only j.h. jacobsen understands how this works,
                # https://github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py
                output = input.permute(0, 2, 3, 1)

                (batch_size, s_height, s_width, s_depth) = output.size()
                d_depth = s_depth * self.block_size_sq
                d_height = s_height // self.block_size

                t_1 = output.split(self.block_size, dim=2)
                stack = [t_t.contiguous().view(batch_size, d_height, d_depth)
                         for t_t in t_1]
                output = torch.stack(stack, 1)
                output = output.permute(0, 2, 1, 3)
                output = output.permute(0, 3, 1, 2)
                return (output.contiguous(),), 0.
            else:
                output = F.conv2d(input, self.downsample_kernel, stride=2, groups=self.channels)
                return (output,), 0.

        else:
            if self.legacy_backend:
                # only j.h. jacobsen understands how this works,
                # https://github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py
                output = input.permute(0, 2, 3, 1)
                (batch_size, d_height, d_width, d_depth) = output.size()
                s_depth = int(d_depth / self.block_size_sq)
                s_width = int(d_width * self.block_size)
                s_height = int(d_height * self.block_size)
                t_1 = output.contiguous().view(batch_size, d_height, d_width,
                                               self.block_size_sq, s_depth)
                spl = t_1.split(self.block_size, 3)
                stack = [t_t.contiguous().view(batch_size, d_height, s_width,
                                               s_depth) for t_t in spl]
                output = torch.stack(stack, 0).transpose(0, 1)
                output = output.permute(0, 2, 1, 3, 4).contiguous()
                output = output.view(batch_size, s_height, s_width, s_depth)
                output = output.permute(0, 3, 1, 2)
                return (output.contiguous(),), 0.
            else:
                output = F.conv_transpose2d(input, self.downsample_kernel,
                                            stride=2, groups=self.channels)
                return (output,), 0.


    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''

        if len(input_dims) != 1:
            raise ValueError("i-RevNet downsampling must have exactly 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError("i-RevNet downsampling can only tranform 2D images"
                             "of the shape CxWxH (channels, width, height)")

        c, w, h = input_dims[0]
        c2, w2, h2 = c * 4, w // 2, h // 2

        if c * h * w != c2 * h2 * w2:
            raise ValueError("Input cannot be cleanly reshaped, most likely because"
                             "the input height or width are an odd number")

        return ((c2, w2, h2),)


class IRevNetUpsampling(IRevNetDownsampling):
    '''The inverted operation of IRevNetDownsampling (see that docstring for details).'''

    def __init__(self, dims_in, dims_c=None, legacy_backend: bool = False):
        '''See docstring of base class (FrEIA.modules.InvertibleModule) for more.
        Args:
          legacy_backend: If True, uses the splitting and concatenating method,
            adapted from
            github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py
            for the use in FrEIA. Is usually slower on GPU.
            If False, uses a 2d strided convolution with a kernel representing
            the downsampling.  Note that the ordering of the output channels
            will be different.  If pixels in each patch in channel 1 
            are a1, b1, ..., and in channel 2 are a2, b2, ...
            Then the output channels will be the following:
            legacy_backend=True: a1, a2, ..., b1, b2, ..., c1, c2, ...
            legacy_backend=False: a1, b1, ..., a2, b2, ..., a3, b3, ...
            (see also order_by_wavelet in module HaarDownsampling)
            Usually this difference is completely irrelevant.
        '''

        # have to initialize with the OUTPUT shape, because everything is
        # inherited from IRevNetDownsampling:
        inv_shape = self.output_dims(dims_in)
        super().__init__(inv_shape, dims_c, legacy_backend=legacy_backend)

    def forward(self, x, c=None, rev=False):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        return super().forward(x, c=None, rev=not rev)

    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''

        if len(input_dims) != 1:
            raise ValueError("i-RevNet downsampling must have exactly 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError("i-RevNet downsampling can only tranform 2D images"
                             "of the shape CxWxH (channels, width, height)")

        c, w, h = input_dims[0]
        c2, w2, h2 = c // 4, w * 2, h * 2

        if c * h * w != c2 * h2 * w2:
            raise ValueError("Input cannot be cleanly reshaped, most likely because"
                             "the input height or width are an odd number")

        return ((c2, w2, h2),)


class HaarDownsampling(InvertibleModule):
    '''Uses Haar wavelets to split each channel into 4 channels, with half the
    width and height dimensions.'''

    def __init__(self, dims_in, dims_c=None,
                 order_by_wavelet: bool = False,
                 rebalance: float = 1.):
        super().__init__()

        self.in_channels = dims_in[0][0]
        self.fac_fwd = 0.5 * rebalance
        self.fac_rev = 0.5 / rebalance
        self.haar_weights = torch.ones(4,1,2,2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

        self.permute = order_by_wavelet
        self.last_jac = None

        if self.permute:
            permutation = []
            for i in range(4):
                permutation += [i+4*j for j in range(self.in_channels)]

            self.perm = torch.LongTensor(permutation)
            self.perm_inv = torch.LongTensor(permutation)

            for i, p in enumerate(self.perm):
                self.perm_inv[p] = i

    def forward(self, x, rev=False):
        if not rev:
            self.last_jac = self.elements / 4 * (np.log(16.) + 4 * np.log(self.fac_fwd))
            out = F.conv2d(x[0], self.haar_weights,
                           bias=None, stride=2, groups=self.in_channels)
            if self.permute:
                return [out[:, self.perm] * self.fac_fwd]
            else:
                return [out * self.fac_fwd]

        else:
            self.last_jac = self.elements / 4 * (np.log(16.) + 4 * np.log(self.fac_rev))
            if self.permute:
                x_perm = x[0][:, self.perm_inv]
            else:
                x_perm = x[0]

            return [F.conv_transpose2d(x_perm * self.fac_rev, self.haar_weights,
                                     bias=None, stride=2, groups=self.in_channels)]

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return self.last_jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c*4, w//2, h//2
        self.elements = c*w*h
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]


class HaarUpsampling(nn.Module):
    '''Uses Haar wavelets to merge 4 channels into one, with double the
    width and height.'''

    def __init__(self, dims_in):
        super().__init__()

        self.in_channels = dims_in[0][0] // 4
        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights *= 0.5
        self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if rev:
            return [F.conv2d(x[0], self.haar_weights,
                             bias=None, stride=2, groups=self.in_channels)]
        else:
            return [F.conv_transpose2d(x[0], self.haar_weights,
                                       bias=None, stride=2,
                                       groups=self.in_channels)]

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c//4, w*2, h*2
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]


class Flatten(nn.Module):
    '''Flattens N-D tensors into 1-D tensors.'''
    def __init__(self, dims_in):
        super().__init__()
        self.size = dims_in[0]

    def forward(self, x, rev=False):
        if not rev:
            return [x[0].view(x[0].shape[0], -1)]
        else:
            return [x[0].view(x[0].shape[0], *self.size)]

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return 0

    def output_dims(self, input_dims):
        return [(int(np.prod(input_dims[0])),)]


class Reshape(nn.Module):
    '''reshapes N-D tensors into target dim tensors.'''
    def __init__(self, dims_in, target_dim):
        super().__init__()
        self.size = dims_in[0]
        self.target_dim = target_dim
        assert int(np.prod(dims_in[0])) == int(np.prod(self.target_dim)), f"Incoming dimensions ({dims_in[0]}) and target_dim ({target_dim}) don't match."

    def forward(self, x, rev=False):
        if not rev:
            return [x[0].reshape(x[0].shape[0], *self.target_dim)]
        else:
            return [x[0].reshape(x[0].shape[0], *self.size)]

    def jacobian(self, x, rev=False):
        return 0.

    def output_dims(self, dim):
        return [self.target_dim]

import warnings

def _deprecated_by(orig_class):
    class deprecated_class(orig_class):
        def __init__(self, *args, **kwargs):

            warnings.warn(F"{self.__class__.__name__} is deprecated and will be removed in the public release. "
                          F"Use {orig_class.__name__} instead.",
                          DeprecationWarning)
            super().__init__(*args, **kwargs)

    return deprecated_class

i_revnet_downsampling = _deprecated_by(IRevNetDownsampling)
i_revnet_upsampling = _deprecated_by(IRevNetUpsampling)
haar_multiplex_layer = _deprecated_by(HaarDownsampling)
haar_restore_layer = _deprecated_by(HaarUpsampling)
flattening_layer = _deprecated_by(Flatten)
reshape_layer = _deprecated_by(Reshape)
