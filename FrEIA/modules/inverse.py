from typing import List, Tuple

from torch import Tensor

from FrEIA.modules import InvertibleModule


class Inverse(InvertibleModule):
    """
    An invertible module that inverses a given module.
    """
    def __init__(self, module: InvertibleModule):
        # Hack as SequenceINN and GraphINN do not work with input/output shape API
        no_output_dims = (
                hasattr(module, "force_tuple_output")
                and not module.force_tuple_output
        )
        if not no_output_dims:
            input_dims = module.output_dims(module.dims_in)
        else:
            try:
                input_dims = module.output_dims(None)
            except TypeError:
                raise NotImplementedError(f"Can't determine output dimensions for {module.__class__}.")
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
