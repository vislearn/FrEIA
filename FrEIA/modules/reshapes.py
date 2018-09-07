import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class i_revnet_downsampling(nn.Module):
    '''The invertible spatial downsampling used in i-RevNet, adapted from
    https://github.com/jhjacobsen/pytorch-i-revnet/blob/master/models/model_utils.py'''

    def __init__(self, dims_in):
        super(i_revnet_downsampling, self).__init__()
        self.block_size = 2
        self.block_size_sq = self.block_size**2

    def forward(self, x, rev=False):
        input = x[0]
        if not rev:
            output = input.permute(0, 2, 3, 1)
            (batch_size, s_height, s_width, s_depth) = output.size()
            d_depth = s_depth * self.block_size_sq
            d_height = int(s_height / self.block_size)
            t_1 = output.split(self.block_size, 2)
            stack = [t_t.contiguous().view(batch_size, d_height, d_depth)
                     for t_t in t_1]
            output = torch.stack(stack, 1)
            output = output.permute(0, 2, 1, 3)
            output = output.permute(0, 3, 1, 2)
            return [output.contiguous()]
            # (own attempt)
            # return torch.cat([
            #         x[:, :,  ::2,  ::2],
            #         x[:, :, 1::2,  ::2],
            #         x[:, :,  ::2, 1::2],
            #         x[:, :, 1::2, 1::2]
            #     ], dim=1)
        else:
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
            return [output.contiguous()]

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c*4, w//2, h//2
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]


class i_revnet_upsampling(i_revnet_downsampling):
    '''Just the exact opposite of the i_revnet_downsampling layer.'''

    def __init__(self, dims_in):
        super(i_revnet_upsampling, self).__init__(dims_in)

    def forward(self, x, rev=False):
        return super(i_revnet_upsampling, self).forward(x, rev=not rev)

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c//4, w*2, h*2
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]


class haar_multiplex_layer(nn.Module):
    '''Uses Haar wavelets to split each channel into 4 channels, with half the
    width and height.'''

    def __init__(self, dims_in, order_by_wavelet=False):
        super(haar_multiplex_layer, self).__init__()

        self.in_channels = dims_in[0][0]
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

        self.permute = order_by_wavelet

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
            out = F.conv2d(x[0], self.haar_weights,
                           bias=None, stride=2, groups=self.in_channels)
            if self.permute:
                return [out[:, self.perm]]
            else:
                return [out]

        else:
            if self.permute:
                x_perm = x[0][:, self.perm_inv]
            else:
                x_perm = x[0]

            return [F.conv_transpose2d(x_perm, self.haar_weights,
                                       bias=None, stride=2,
                                       groups=self.in_channels)]

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c*4, w//2, h//2
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return [(c2, w2, h2)]


class haar_restore_layer(nn.Module):
    '''Uses Haar wavelets to merge 4 channels into one, with double the
    width and height.'''

    def __init__(self, dims_in):
        super(haar_restore_layer, self).__init__()

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


class flattening_layer(nn.Module):
    '''Flattens N-D tensors into 1-D tensors.'''
    def __init__(self, dims_in):
        super(flattening_layer, self).__init__()
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
