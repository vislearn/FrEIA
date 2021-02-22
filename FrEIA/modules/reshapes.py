from . import InvertibleModule

from warnings import warn
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IRevNetDownsampling(InvertibleModule):
    '''The invertible spatial downsampling used in i-RevNet.
    Each group of four neighboring pixels is reordered into one pixel with four times
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
            the downsampling. Note that the ordering of the output channels
            will be different. If pixels in each patch in channel 1
            are ``a1, b1,...``, and in channel 2 are ``a2, b2,...``
            Then the output channels will be the following:

            ``legacy_backend=True: a1, a2, ..., b1, b2, ..., c1, c2, ...``

            ``legacy_backend=False: a1, b1, ..., a2, b2, ..., a3, b3, ...``

            (see also order_by_wavelet in module HaarDownsampling)
            Generally this difference is completely irrelevant,
            unless a certaint subset of pixels or channels is supposed to be
            split off or extracted.
        '''
        super().__init__(dims_in, dims_c)

        self.channels = dims_in[0][0]
        self.block_size = 2
        self.block_size_sq = self.block_size**2
        self.legacy_backend = legacy_backend

        if not self.legacy_backend:
            # this kernel represents the reshape:
            # it applies to 2x2 patches (stride 2), and transforms each
            # input channel to 4 channels.
            # The input value is transferred wherever the kernel is 1.
            # (hence the indexing pattern 00, 01, 10, 11 represents the
            # checkerboard.
            # For the upsampling, a transposed convolution is used for the
            # opposite effect.

            self.downsample_kernel = torch.zeros(4, 1, 2, 2)

            self.downsample_kernel[0, 0, 0, 0] = 1
            self.downsample_kernel[1, 0, 0, 1] = 1
            self.downsample_kernel[2, 0, 1, 0] = 1
            self.downsample_kernel[3, 0, 1, 1] = 1

            self.downsample_kernel = torch.cat([self.downsample_kernel] * self.channels, 0)
            self.downsample_kernel = nn.Parameter(self.downsample_kernel)
            self.downsample_kernel.requires_grad = False

    def forward(self, x, c=None, jac=True, rev=False):
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
            raise ValueError("i-RevNet downsampling can only transform 2D images"
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
            If False, uses a 2d strided transposed convolution with a representing
            the downsampling. Note that the expected ordering of the input channels
            will be different. If pixels in each output patch in channel 1
            are ``a1, b1,...``, and in channel 2 are ``a2, b2,...``
            Then the expected input channels are be the following:

            ``legacy_backend=True: a1, a2, ..., b1, b2, ..., c1, c2, ...``

            ``legacy_backend=False: a1, b1, ..., a2, b2, ..., a3, b3, ...``

            (see also order_by_wavelet in module HaarDownsampling)
            Generally this difference is completely irrelevant,
            unless a certaint subset of pixels or channels is supposed to be
            split off or extracted.
        '''

        # have to initialize with the OUTPUT shape, because everything is
        # inherited from IRevNetDownsampling:
        inv_shape = self.output_dims(dims_in)
        super().__init__(inv_shape, dims_c, legacy_backend=legacy_backend)

    def forward(self, x, c=None, jac=True, rev=False):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        return super().forward(x, c=None, rev=not rev)

    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''

        if len(input_dims) != 1:
            raise ValueError("i-RevNet downsampling must have exactly 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError("i-RevNet downsampling can only transform 2D images"
                             "of the shape cxwxh (channels, width, height)")

        c, w, h = input_dims[0]
        c2, w2, h2 = c // 4, w * 2, h * 2

        if c * h * w != c2 * h2 * w2:
            raise ValueError("input cannot be cleanly reshaped, most likely because"
                             "the input height or width are an odd number")

        return ((c2, w2, h2),)


class HaarDownsampling(InvertibleModule):
    '''Uses Haar wavelets to split each channel into 4 channels, with half the
    width and height dimensions.'''

    def __init__(self, dims_in, dims_c = None,
                 order_by_wavelet: bool = False,
                 rebalance: float = 1.):
        '''See docstring of base class (FrEIA.modules.InvertibleModule) for more.

        Args:
          order_by_wavelet: Whether to group the output by original channels or
            by wavelet. I.e. if the average, vertical, horizontal and diagonal
            wavelets for channel 1 are ``a1, v1, h1, d1``, those for channel 2 are
            ``a2, v2, h2, d2``, etc, then the output channels will be structured as
            follows:

            set to ``True: a1, a2, ..., v1, v2, ..., h1, h2, ..., d1, d2, ...``

            set to ``False: a1, v1, h1, d1, a2, v2, h2, d2, ...``

            The ``True`` option is slightly slower to compute than the ``False`` option.
            The option is useful if e.g. the average channels should be split
            off by a FrEIA.modules.Split. Then, setting ``order_by_wavelet=True``
            allows to split off the first quarter of channels to isolate the
            average wavelets only.
          rebalance: Must be !=0. There exist different conventions how to define
            the Haar wavelets. The wavelet components in the forward direction
            are multiplied with this factor, and those in the inverse direction
            are adjusted accordingly, so that the module as a whole is
            invertible.  Stability of the network may be increased for rebalance
            < 1 (e.g. 0.5).
        '''
        super().__init__(dims_in, dims_c)

        if rebalance == 0:
            raise ValueError("'rebalance' argument must be != 0.")

        self.in_channels = dims_in[0][0]

        # self.jac_{fwd,rev} is the log Jacobian determinant for a single pixel
        # in a single channel computed explicitly from the matrix below.

        self.fac_fwd = 0.5 * rebalance
        self.jac_fwd = (np.log(16.) + 4 * np.log(self.fac_fwd)) / 4.

        self.fac_rev = 0.5 / rebalance
        self.jac_rev = (np.log(16.) + 4 * np.log(self.fac_rev)) / 4.

        # See https://en.wikipedia.org/wiki/Haar_wavelet#Haar_matrix
        # for an explanation of how this weight matrix comes about
        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

        # for 'order_by_wavelet', we just perform the channel-wise wavelet
        # transform as usual, and then permute the channels into the correct
        # order afterward (hence 'self.permute')
        self.permute = order_by_wavelet

        if self.permute:
            permutation = []
            for i in range(4):
                permutation += [i + 4 * j for j in range(self.in_channels)]

            self.perm = torch.LongTensor(permutation)
            self.perm_inv = torch.LongTensor(permutation)

            # clever trick to invert a permutation
            for i, p in enumerate(self.perm):
                self.perm_inv[p] = i

    def forward(self, x, c=None, jac=True, rev=False):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''

        inp = x[0]
        #number total entries except for batch dimension:
        ndims = inp[0].numel()

        if not rev:
            jac = ndims * self.jac_fwd
            out = F.conv2d(inp, self.haar_weights,
                           bias=None, stride=2, groups=self.in_channels)

            if self.permute:
                return (out[:, self.perm] * self.fac_fwd,), jac
            else:
                return (out * self.fac_fwd,), jac

        else:
            jac = ndims * self.jac_rev
            if self.permute:
                x_perm = inp[:, self.perm_inv]
            else:
                x_perm = inp

            x_perm *= self.fac_rev
            out = F.conv_transpose2d(x_perm, self.haar_weights, stride=2, groups=self.in_channels)

            return (out,), jac

    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''

        if len(input_dims) != 1:
            raise ValueError("HaarDownsampling must have exactly 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError("HaarDownsampling can only transform 2D images"
                             "of the shape CxWxH (channels, width, height)")

        c, w, h = input_dims[0]
        c2, w2, h2 = c * 4, w // 2, h // 2

        if c * h * w != c2 * h2 * w2:
            raise ValueError("Input cannot be cleanly reshaped, most likely because"
                             "the input height or width are an odd number")

        return ((c2, w2, h2),)


class HaarUpsampling(HaarDownsampling):
    '''The inverted operation of HaarDownsampling (see that docstring for details).'''

    def __init__(self, dims_in, dims_c = None,
                 order_by_wavelet: bool = False,
                 rebalance: float = 1.):
        '''See docstring of base class (FrEIA.modules.InvertibleModule) for more.

        Args:
          order_by_wavelet: Expected grouping of the input channels by wavelet or
            by output channel. I.e. if the average, vertical, horizontal and diagonal
            wavelets for channel 1 are ``a1, v1, h1, d1``, those for channel 2 are
            ``a2, v2, h2, d2``, etc, then the input channels are taken as follows:

            set to ``True: a1, a2, ..., v1, v2, ..., h1, h2, ..., d1, d2, ...``

            set to ``False: a1, v1, h1, d1, a2, v2, h2, d2, ...``

            The ``True`` option is slightly slower to compute than the ``False`` option.
            The option is useful if e.g. the input has been concatentated from average
            channels and the higher-frequency channels. Then, setting
            ``order_by_wavelet=True`` allows to split off the first quarter of
            channels to isolate the average wavelets only.
          rebalance: Must be !=0. There exist different conventions how to define
            the Haar wavelets. The wavelet components in the forward direction
            are multiplied with this factor, and those in the inverse direction
            are adjusted accordingly, so that the module as a whole is
            invertible.  Stability of the network may be increased for rebalance
            < 1 (e.g. 0.5).
        '''
        inv_shape = self.output_dims(dims_in)
        super().__init__(inv_shape, dims_c, order_by_wavelet, rebalance)

    def forward(self, x, c=None, jac=True, rev=False):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        return super().forward(x, c=None, rev=not rev)

    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''

        if len(input_dims) != 1:
            raise ValueError("i-revnet downsampling must have exactly 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError("i-revnet downsampling can only tranform 2d images"
                             "of the shape cxwxh (channels, width, height)")

        c, w, h = input_dims[0]
        c2, w2, h2 = c // 4, w * 2, h * 2

        if c * h * w != c2 * h2 * w2:
            raise ValueError("input cannot be cleanly reshaped, most likely because"
                             "the input height or width are an odd number")

        return ((c2, w2, h2),)


class Flatten(InvertibleModule):
    '''Flattens N-D tensors into 1-D tensors.'''

    def __init__(self, dims_in, dims_c=None):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        super().__init__(dims_in, dims_c)

        if len(dims_in) != 1:
            raise ValueError("Flattening must have exactly 1 input")

        self.input_shape = dims_in[0]
        self.output_shape = (int(np.prod(dims_in[0])),)

    def forward(self, x, c=None, jac=True, rev=False):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        if not rev:
            return (x[0].view(x[0].shape[0], -1),), 0.
        else:
            return (x[0].view(x[0].shape[0], *self.input_shape),), 0.

    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        return (self.output_shape,)


class Reshape(InvertibleModule):
    '''Reshapes N-D tensors into target dim tensors. Note that the reshape resulting from
    e.g. (3, 32, 32) -> (12, 16, 16) will not necessarily be spatially sensible.
    See ``IRevNetDownsampling``, ``IRevNetUpsampling``, ``HaarDownsampling``, 
    ``HaarUpsampling`` for spatially meaningful reshaping operations.'''

    def __init__(self, dims_in, dims_c=None, output_dims: Iterable[int] = None, target_dim = None):
        '''See docstring of base class (FrEIA.modules.InvertibleModule) for more.

        Args:
          output_dims: The shape the reshaped output is supposed to have (not
            including batch dimension)
          target_dim: Deprecated name for output_dims
        '''
        super().__init__(dims_in, dims_c)

        if target_dim is not None:
            warn("Use the new name for the 'target_dim' argument: 'output_dims'"
                 "the 'target_dim' argument will be removed in the next version")
            output_dims = target_dim

        if output_dims is None:
            raise ValueError("Please specify the desired output shape")

        self.size = dims_in[0]
        self.target_dim = output_dims

        if len(dims_in) != 1:
            raise ValueError("Reshape must have exactly 1 input")
        if int(np.prod(dims_in[0])) != int(np.prod(self.target_dim)):
            raise ValueError(f"Incoming dimensions {dims_in[0]} and target_dim"
                             f"{self.target_dim} don't match."
                             "Must have same number of elements for invertibility")

    def forward(self, x, c=None, jac=True, rev=False):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''

        if not rev:
            return (x[0].reshape(x[0].shape[0], *self.target_dim),), 0.
        else:
            return (x[0].reshape(x[0].shape[0], *self.size),), 0.

    def output_dims(self, dim):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        return (self.target_dim,)
