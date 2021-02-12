import warnings
from typing import Union, Iterable, Tuple

from torch import Tensor

from .graph_inn import GraphINN


class ReversibleGraphNet(GraphINN):
    def __init__(self, node_list, ind_in=None, ind_out=None, verbose=True,
                 force_tuple_output=False):
        warnings.warn("ReversibleGraphNet is deprecated in favour of GraphINN. "
                      "It will be removed in the next version of FrEIA.",
                      DeprecationWarning)
        if ind_in is not None:
            raise ValueError(
                "ReversibleGraphNet's ind_in was removed in FrEIA v0.3.0. "
                "Please use InputNodes and switch to GraphINN."
            )
        if ind_out is not None:
            raise ValueError(
                "ReversibleGraphNet's ind_out was removed in FrEIA v0.3.0. "
                "Please use OutputNodes and switch to GraphINN."
            )
        super().__init__(node_list, verbose=verbose,
                         force_tuple_output=force_tuple_output)

    def forward(self, x_or_z: Union[Tensor, Iterable[Tensor]],
                c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True,
                intermediate_outputs: bool = False)\
            -> Tuple[Tuple[Tensor], Tensor]:
        warnings.warn("ReversibleGraphNet's forward() now "
                      "returns a tuple (output, jacobian). "
                      "It will be removed in the next version of FrEIA.",
                      DeprecationWarning)
        return super().forward(x_or_z, c, rev, jac, intermediate_outputs)
