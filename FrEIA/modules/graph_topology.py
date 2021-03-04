from . import InvertibleModule

import warnings
from copy import deepcopy
from typing import Sequence, Union

import torch


class Split(InvertibleModule):
    """Invertible split operation.

    Splits the incoming tensor along the given dimension, and returns a list of
    separate output tensors. The inverse is the corresponding merge operation.

    """

    def __init__(self,
                 dims_in: Sequence[Sequence[int]],
                 section_sizes: Union[int, Sequence[int]] = None,
                 n_sections: int = 2,
                 dim: int = 0,
     ):
        """Inits the Split module with the attributes described above and
        checks that split sizes and dimensionality are compatible.

        Args:
          dims_in:
            A list of tuples containing the non-batch dimensionality of all
            incoming tensors. Handled automatically during compute graph setup.
            Split only takes one input tensor.
          section_sizes:
            If set, takes precedence over ``n_sections`` and behaves like the
            argument in torch.split(), except when a list of section sizes is given
            that doesn't add up to the size of ``dim``, an additional split section is
            created to take the slack. Defaults to None.
          n_sections:
            If ``section_sizes`` is None, the tensor is split into ``n_sections``
            parts of equal size or close to it. This mode behaves like
            ``numpy.array_split()``. Defaults to 2, i.e. splitting the data into two
            equal halves.
          dim:
            Index of the dimension along which to split, not counting the batch
            dimension. Defaults to 0, i.e. the channel dimension in structured data.
        """
        super().__init__(dims_in)

        # Size and dimensionality checks
        assert len(dims_in) == 1, "Split layer takes exactly one input tensor"
        assert len(dims_in[0]) >= dim, "Split dimension index out of range"
        self.dim = dim
        l_dim = dims_in[0][dim]

        if section_sizes is None:
            assert 2 <= n_sections, "'n_sections' must be a least 2"
            if l_dim % n_sections != 0:
                warnings.warn('Split will create sections of unequal size')
            self.split_size_or_sections = (
                [l_dim//n_sections + 1] * (l_dim%n_sections) +
                [l_dim//n_sections] * (n_sections - l_dim%n_sections))
        else:
            if isinstance(section_sizes, int):
                assert section_sizes < l_dim, "'section_sizes' too large"
            else:
                assert isinstance(section_sizes, (list, tuple)), \
                    "'section_sizes' must be either int or list/tuple of int"
                assert sum(section_sizes) <= l_dim, "'section_sizes' too large"
                if sum(section_sizes) < l_dim:
                    warnings.warn("'section_sizes' too small, adding additional section")
                    section_sizes = list(section_sizes).append(l_dim - sum(section_sizes))
            self.split_size_or_sections = section_sizes

    def forward(self, x, rev=False, jac=True):
        """See super class InvertibleModule.
        Jacobian log-det of splitting is always zero."""
        if rev:
            return [torch.cat(x, dim=self.dim+1)], 0
        else:
            return torch.split(x[0], self.split_size_or_sections,
                               dim=self.dim+1), 0

    def output_dims(self, input_dims):
        """See super class InvertibleModule."""
        assert len(input_dims) == 1, "Split layer takes exactly one input tensor"
        # Assemble dims of all resulting outputs
        return [tuple(input_dims[0][j] if (j != self.dim) else section_size
                      for j in range(len(input_dims[0])))
                for section_size in self.split_size_or_sections]



class Concat(InvertibleModule):
    """Invertible merge operation.

    Concatenates a list of incoming tensors along a given dimension and passes
    on the result. Inverse is the corresponding split operation.
    """

    def __init__(self,
                 dims_in: Sequence[Sequence[int]],
                 dim: int = 0,
     ):
        """Inits the Concat module with the attributes described above and
        checks that all dimensions are compatible.

        Args:
          dims_in:
            A list of tuples containing the non-batch dimensionality of all
            incoming tensors. Handled automatically during compute graph setup.
            Dimensionality of incoming tensors must be identical, except in the
            merge dimension ``dim``. Concat only makes sense with multiple input
            tensors.
          dim:
            Index of the dimension along which to concatenate, not counting the
            batch dimension. Defaults to 0, i.e. the channel dimension in structured
            data.
        """
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

    def forward(self, x, rev=False, jac=True):
        """See super class InvertibleModule.
        Jacobian log-det of concatenation is always zero."""
        if rev:
            return torch.split(x[0], self.split_size_or_sections,
                               dim=self.dim+1), 0
        else:
            return [torch.cat(x, dim=self.dim+1)], 0

    def output_dims(self, input_dims):
        """See super class InvertibleModule."""
        assert len(input_dims) > 1, ("Concatenation only makes sense for "
                                     "multiple inputs")
        output_dims = deepcopy(list(input_dims[0]))
        output_dims[self.dim] = sum(input_dim[self.dim]
                                    for input_dim in input_dims)
        return [tuple(output_dims)]


def _deprecated_by(orig_class):
    class deprecated_class(orig_class):
        def __init__(self, *args, **kwargs):
            warnings.warn(F"{self.__class__.__name__} is deprecated and will be removed in the public release. "
                          F"Use {orig_class.__name__} instead.",
                          DeprecationWarning)
            super().__init__(*args, **kwargs)

    return deprecated_class

_depr_docstring = "This class is deprecated and replaced by ``{}``"

Split1D = _deprecated_by(Split)
Split1D.__doc__ = _depr_docstring.format(Split.__name__)

SplitChannel = _deprecated_by(Split)
SplitChannel.__doc__ = _depr_docstring.format(Split.__name__)

Concat1d = _deprecated_by(Concat)
Concat1d.__doc__ = _depr_docstring.format(Concat.__name__)

ConcatChannel = _deprecated_by(Concat)
ConcatChannel.__doc__ = _depr_docstring.format(Concat.__name__)
