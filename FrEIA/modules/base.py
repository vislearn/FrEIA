from typing import Tuple, Iterable, List

import torch.nn as nn
from torch import Tensor


class InvertibleModule(nn.Module):
    """
    Base class for all invertible modules in FrEIA.

    Given `module`, an instance of some InvertibleOperator.
    This `module` shall be invertible in its input dimensions,
    so that the input can be recovered by applying the module
    in backwards mode (`rev=True`), not to be confused with
    `pytorch.backward()` which computes the gradient of an operation.
    ```
    x = torch.randn(BATCH_SIZE, DIM_COUNT)
    c = torch.randn(BATCH_SIZE, CONDITION_DIM)

    # Forward mode
    z, jac = module([x], [c], jac=True)

    # Backward mode
    x_rev, jac_rev = module(z, [c], rev=True, jac=True)
    ```
    The `module` returns $\log \det J = \log \det \frac{\partial f}{\partial x}$
    of the operation in forward mode, and
    $-\log \det J = \log \det \frac{\partial f^{-1}}{\partial z} = -\log \det \frac{\partial f}{\partial x}$
    in backward mode (`rev=True`).

    Then, `torch.allclose(x, x_rev[0]) == True` and `jac == -jac_rev`.
    """

    def __init__(self, dims_in: Iterable[Tuple[int]],
                 dims_c: Iterable[Tuple[int]] = None):
        """
        Parameters:
            dims_in: list of tuples specifying the shape of the inputs to this
                     operator: dims_in = [shape_x_0, shape_x_1, ...]
            dims_c:  list of tuples specifying the shape of the conditions to
                     this operator.
        """
        super().__init__()
        if dims_c is None:
            dims_c = []
        self.dims_in = list(dims_in)
        self.dims_c = list(dims_c)

    def forward(self, x_or_z: Iterable[Tensor], c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = True) \
            -> Tuple[Tuple[Tensor], Tensor]:
        """
        Perform a forward (default, `rev=False`) or backward pass (`rev=True`)
        through this module/operator.

        *Note to implementers:*
        - Subclasses MUST return a Jacobian when jac=True, but CAN return a
          valid Jacobian when jac=False (not punished). The latter is only recommended
          if the computation of the Jacobian is trivial.
        - Subclasses MUST follow the convention that the returned Jacobian be
          consistent with the evaluation direction. Let's make this more precise:
          Let $f$ be the function that the subclass represents. Then:
          $$$
          J = \log \det \frac{\partial f}{\partial x} \\
          -J = \log \det \frac{\partial f^{-1}}{\partial z}.
          $$$
          Any subclass MUST return $J$ for forward evaluation (`rev=False`),
          and $-J$ for backward evaluation (`rev=True`).

        Parameters:
            x_or_z: input data (array-like of one or more tensors)
            c:      conditioning data (array-like of none or more tensors)
            rev:    perform backward pass
            jac:    return Jacobian associated to the direction
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide forward(...) method")

    def jacobian(self, *args, **kwargs):
        raise DeprecationWarning("module.jacobian(...) is deprecated. "
                                 "module.forward(..., jac=True) returns a "
                                 "tuple (out, jacobian) now.")

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide output_dims(...)")
