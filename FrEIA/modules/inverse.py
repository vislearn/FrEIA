from typing import List, Tuple

from torch import Tensor

from FrEIA.modules import InvertibleModule
from FrEIA.utils import output_dims_compatible


class Inverse(InvertibleModule):
    """
    An invertible module that inverses a given module.
    """
    def __init__(self, module: InvertibleModule):
        # Hack as SequenceINN and GraphINN do not work with input/output shape API
        input_dims = output_dims_compatible(module)
        super().__init__(input_dims, module.dims_c)
        self.module = module

    @property
    def force_tuple_output(self):
        try:
            return self.module.force_tuple_output
        except AttributeError:
            return True

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        return self.module.dims_in

    def forward(self, *args,
                rev: bool = False, **kwargs) -> Tuple[Tuple[Tensor], Tensor]:
        return self.module(*args, rev=not rev, **kwargs)
