from typing import Tuple, Iterable, Union

import torch.nn as nn
from torch import Tensor, is_tensor


class InvertibleOperator(nn.Module):
    def __init__(self, dims_in: Tuple[int], dims_c: Tuple[int] = None):
        super().__init__()
        if dims_c is None:
            dims_c = []
        self.dims_in = dims_in
        self.dims_c = dims_c

    def forward(self, x: Union[Iterable[Tensor], Tensor], c: Union[Iterable[Tensor], Tensor], rev=False, jac=True):
        # Tupelize input
        is_tensor_input = is_tensor(x)
        if is_tensor_input:
            x = (x,)
        if not is_tensor(c):
            c = (c,)

        # Compute
        z, jac = self.compute(x, c, rev=rev, jac=jac)

        # Check returned types -- todo unit test?
        assert isinstance(z, tuple), f"{self.__class__.__name__}.compute(...) returned non-tuple vector"
        if jac:
            assert is_tensor(jac), f"{self.__class__.__name__}.compute(...) returned non-Tensor Jacobian"

        # Ensure output follows input
        if is_tensor_input:
            return z[0], jac
        else:
            return z, jac

    def compute(self, x: Iterable[Tensor], c: Iterable[Tensor], rev: bool, jac: bool) -> Tuple[Tuple[Tensor], Tensor]:
        raise NotImplementedError(f"{self.__class__.__name__} does not provide compute(...) method")
