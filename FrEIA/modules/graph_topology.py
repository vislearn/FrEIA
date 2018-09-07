from copy import deepcopy

import torch
import torch.nn as nn


class channel_split_layer(nn.Module):
    '''Splits along channels to produce two separate outputs (for skip connections
    and such).'''
    def __init__(self, dims_in):
        super(channel_split_layer, self).__init__()
        assert len(dims_in) == 1, "Use channel_merge_layer instead"
        self.channels = dims_in[0][0]

    def forward(self, x, rev=False):
        if rev:
            return [torch.cat(x, dim=1)]
        else:
            return [x[0][:, :self.channels//2], x[0][:, self.channels//2:]]

    def jacobian(self, x, rev=False):
        # TODO batch size
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Use channel_merge_layer instead"
        return [[input_dims[0][0]//2, *input_dims[0][1:]],
                [input_dims[0][0] - input_dims[0][0]//2, *input_dims[0][1:]]]


class channel_merge_layer(nn.Module):
    '''Merges along channels from two separate inputs, to one output
    (for skip connections etc.)'''
    def __init__(self, dims_in):
        super(channel_merge_layer, self).__init__()
        assert len(dims_in) == 2, "Can only merge 2 inputs"
        self.ch1 = dims_in[0][0]
        self.ch2 = dims_in[1][0]

    def forward(self, x, rev=False):
        if rev:
            return [x[0][:, :self.ch1], x[0][:, self.ch1:]]
        else:
            return [torch.cat(x, dim=1)]

    def jacobian(self, x, rev=False):
        # TODO batch size
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 2, "Can only merge 2 inputs"

        return [[input_dims[0][0] + input_dims[1][0], *input_dims[0][1:]]]


class split_layer(nn.Module):
    '''Splits along given dimension to produce list of separate outputs with
    given size.'''
    def __init__(self, dims_in, split_size_or_sections, dim):
        super(split_layer, self).__init__()
        assert len(dims_in) == 1, "Split layer takes exactly one input tensor"
        assert len(dims_in[0]) >= dim, "Split dimension index out of range"
        if isinstance(split_size_or_sections, int):
            assert dims_in[0][dim] % split_size_or_sections == 0, (
                "Tensor size not divisible by split size"
            )
        else:
            assert isinstance(split_size_or_sections, (list, tuple)), (
                "'split_size_or_sections' must be either int or "
                "list/tuple of int"
            )

            assert dims_in[0][dim] == sum(split_size_or_sections), (
                "Tensor size doesn't match sum of split sections "
                f"({dims_in[0][dim]} vs {split_size_or_sections})"
            )
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, x, rev=False):
        if rev:
            return [torch.cat(x, dim=self.dim+1)]
        else:
            return torch.split(x[0], self.split_size_or_sections,
                               dim=self.dim+1)

    def jacobian(self, x, rev=False):
        # TODO batch size
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, ("Split layer takes exactly one input "
                                      "tensor")
        if isinstance(self.split_size_or_sections, int):
            self.split_size_or_sections = (
                [self.split_size_or_sections]
                * (input_dims[0][self.dim] // self.split_size_or_sections)
            )
        return [[input_dims[0][j] if j != self.dim else split_size
                 for j in range(len(input_dims[0]))]
                for split_size in self.split_size_or_sections]


class cat_layer(nn.Module):
    '''Merge multiple tensors along given dimension.'''
    def __init__(self, dims_in, dim):
        super(cat_layer, self).__init__()
        assert len(dims_in) > 1, ("Concatenation only makes sense for "
                                  "multiple inputs")
        assert len(dims_in[0]) >= dim, "Merge dimension index out of range"
        assert all(len(dims_in[i]) == len(dims_in[0])
                   for i in range(len(dims_in))), (
                           "All input tensors must have same number of "
                           "dimensions"
                   )
        assert all(dims_in[i][j] == dims_in[0][j] for i in range(len(dims_in))
                   for j in range(len(dims_in[i])) if j != dim), (
                           "All input tensor dimensions except merge "
                           "dimension must be identical"
                   )
        self.dim = dim
        self.split_size_or_sections = [dims_in[i][dim]
                                       for i in range(len(dims_in))]

    def forward(self, x, rev=False):
        if rev:
            return torch.split(x[0], self.split_size_or_sections,
                               dim=self.dim+1)
        else:
            return [torch.cat(x, dim=self.dim+1)]

    def jacobian(self, x, rev=False):
        # TODO batch size
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) > 1, ("Concatenation only makes sense for "
                                     "multiple inputs")
        output_dims = deepcopy(list(input_dims[0]))
        output_dims[self.dim] = sum(input_dim[self.dim]
                                    for input_dim in input_dims)
        return [output_dims]
