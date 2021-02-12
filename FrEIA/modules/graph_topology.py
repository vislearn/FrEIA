from copy import deepcopy

import torch
import torch.nn as nn

from FrEIA.modules import InvertibleModule


class Split(InvertibleModule):
    """Invertible split operation.

    Splits the incoming tensor along the given dimension, and returns a list of
    separate output tensors. The inverse is the corresponding merge operation.

    Attributes:
      dims_in:
        A list of tuples containing the non-batch dimensionality of all
        incoming tensors. Handled automatically during compute graph setup.
        Split only takes one input tensor.
      split_size_or_sections:
        Same behavior as in `torch.split()`. Either an integer that is a valid
        divisor of the size of `dim`, or a list of integers that add up to the
        size of `dim`. Defaults to `2`, i.e. splitting the input in two halves
        of equal size.
      dim:
        Index of the dimension along which to split, not counting the batch
        dimension. Defaults to 0, i.e. the channel dimension in structured data.
    """

    def __init__(self, dims_in, split_size_or_sections=2, dim=0):
        """Inits the Split module with the attributes described above and
        checks that split sizes and dimensionality are compatible."""
        super().__init__(dims_in)
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

    def forward(self, x_or_z, c=None, rev=False, jac=True):
        """See super class InvertibleModule."""
        if rev:
            return [torch.cat(x_or_z, dim=self.dim+1)], \
                   torch.zeros(x_or_z[0].shape[0])
        else:
            return torch.split(x_or_z[0], self.split_size_or_sections,
                               dim=self.dim+1), torch.zeros(x_or_z[0].shape[0])

    def output_dims(self, input_dims):
        """See super class InvertibleModule."""
        assert len(input_dims) == 1, ("Split layer takes exactly one input "
                                      "tensor")
        # Turn number of sections into list of section sizes
        if isinstance(self.split_size_or_sections, int):
            self.split_size_or_sections = (
                [input_dims[0][self.dim] // self.split_size_or_sections]
                * self.split_size_or_sections
            )
        # Assemble dims of all resulting outputs
        return [tuple(input_dims[0][j] if (j != self.dim) else split_size
                      for j in range(len(input_dims[0])))
                for split_size in self.split_size_or_sections]


class Concat(InvertibleModule):
    """Invertible merge operation.

    Concatenates a list of incoming tensors along a given dimension and passes
    on the result. Inverse is the corresponding split operation.

    Attributes:
      dims_in:
        A list of tuples containing the non-batch dimensionality of all
        incoming tensors. Handled automatically during compute graph setup.
        Dimensionality of incoming tensors must be identical, except in the
        merge dimension `dim`. Concat only makes sense with multiple input
        tensors.
      dim:
        Index of the dimension along which to concatenate, not counting the
        batch dimension. Defaults to 0, i.e. the channel dimension in structured
        data.
    """

    def __init__(self, dims_in, dim=0):
        """Inits the Concat module with the attributes described above and
        checks that all dimensions are compatible."""
        super().__init__(dims_in)
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
        """See super class InvertibleModule."""
        if rev:
            return torch.split(x[0], self.split_size_or_sections,
                               dim=self.dim+1), torch.zeros(x[0].shape[0])
        else:
            return [torch.cat(x, dim=self.dim+1)], torch.zeros(x[0].shape[0])

    def output_dims(self, input_dims):
        """See super class InvertibleModule."""
        assert len(input_dims) > 1, ("Concatenation only makes sense for "
                                     "multiple inputs")
        output_dims = deepcopy(list(input_dims[0]))
        output_dims[self.dim] = sum(input_dim[self.dim]
                                    for input_dim in input_dims)
        return [tuple(output_dims)]



import warnings

def _deprecated_by(orig_class):
    class deprecated_class(orig_class):
        def __init__(self, *args, **kwargs):

            warnings.warn(F"{self.__class__.__name__} is deprecated and will be removed in the public release. "
                          F"Use {orig_class.__name__} instead.",
                          DeprecationWarning)
            super().__init__(*args, **kwargs)

    return deprecated_class

channel_split_layer = _deprecated_by(Split)
split_layer = _deprecated_by(Split)
Split1D = _deprecated_by(Split)
SplitChannel = _deprecated_by(Split)
channel_merge_layer = _deprecated_by(Concat)
cat_layer = _deprecated_by(Concat)
Concat1d = _deprecated_by(Concat)
ConcatChannel = _deprecated_by(Concat)
