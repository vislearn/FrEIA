from typing import List, Tuple

from torch import Tensor

from FrEIA.modules import InvertibleModule


class Inverse(InvertibleModule):
    """
    An invertible module that inverses a given module.
    """
    def __init__(self, module: InvertibleModule):
        super().__init__(module.output_dims(module.dims_in), module.dims_c)
        self.module = module

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        return self.module.dims_in

    def forward(self, *args,
                rev: bool = False, **kwargs) -> Tuple[Tuple[Tensor], Tensor]:
        return self.module(*args, rev=not rev, **kwargs)
