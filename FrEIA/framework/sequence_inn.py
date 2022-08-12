from typing import Iterable, Tuple, List

import torch.nn as nn
import torch
from torch import Tensor

from FrEIA.modules import InvertibleModule


class SequenceINN(InvertibleModule):
    """
    Simpler than FrEIA.framework.GraphINN:
    Only supports a sequential series of modules (no splitting, merging,
    branching off).
    Has an append() method, to add new blocks in a more simple way than the
    computation-graph based approach of GraphINN. For example:

    .. code-block:: python

       inn = SequenceINN(channels, dims_H, dims_W)

       for i in range(n_blocks):
           inn.append(FrEIA.modules.AllInOneBlock, clamp=2.0, permute_soft=True)
       inn.append(FrEIA.modules.HaarDownsampling)
       # and so on
    """

    def __init__(self, *dims: int, force_tuple_output=False):
        super().__init__([dims])

        self.shapes = [tuple(dims)]
        self.conditions = []
        self.module_list = nn.ModuleList()

        self.force_tuple_output = force_tuple_output

    def append(self, module_class, cond=None, cond_shape=None, **kwargs):
        """
        Append a reversible block from FrEIA.modules to the network.

        Args:
          module_class: Class from FrEIA.modules.
          cond (int): index of which condition to use (conditions will be passed as list to forward()).
            Conditioning nodes are not needed for SequenceINN.
          cond_shape (tuple[int]): the shape of the condition tensor.
          **kwargs: Further keyword arguments that are passed to the constructor of module_class (see example).
        """

        dims_in = [self.shapes[-1]]
        self.conditions.append(cond)

        if cond is not None:
            kwargs['dims_c'] = [cond_shape]

        if isinstance(module_class, InvertibleModule):
            module = module_class
            if module.dims_in != dims_in:
                raise ValueError(
                    f"You passed an instance of {module.__class__} to "
                    f"SequenceINN which expects a {module.dims_in} input, "
                    f"but the output of the previous layer is of shape "
                    f"{dims_in}."
                )
        else:
            module = module_class(dims_in, **kwargs)
        self.module_list.append(module)
        output_dims = module.output_dims(dims_in)
        if len(output_dims) != 1:
            raise ValueError(
                f"Module of type {module.__class__} has more than one output: "
                f"{output_dims}"
            )
        self.shapes.append(output_dims[0])

    def __getitem__(self, item):
        return self.module_list.__getitem__(item)

    def __len__(self):
        return self.module_list.__len__()

    def __iter__(self):
        return self.module_list.__iter__()

    def output_dims(self, input_dims: List[Tuple[int]] = None) \
            -> List[Tuple[int]]:
        """
        Extends the definition in InvertibleModule to also return the output
        dimension when
        """
        if input_dims is not None:
            if self.force_tuple_output:
                if input_dims != self.shapes[0]:
                    raise ValueError(f"Passed input shapes {input_dims!r} do "
                                     f"not match with those passed in the "
                                     f"construction of the SequenceINN "
                                     f"{self.shapes[0]}")
            else:
                raise ValueError("You can only call output_dims on a "
                                 "SequenceINN when setting "
                                 "force_tuple_output=True.")
        return [self.shapes[-1]]

    def forward(self, x_or_z: Tensor, c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Executes the sequential INN in forward or inverse (rev=True) direction.

        Args:
            x_or_z: input tensor (in contrast to GraphINN, a list of
                    tensors is not supported, as SequenceINN only has
                    one input).
            c: list of conditions.
            rev: whether to compute the network forward or reversed.
            jac: whether to compute the log jacobian

        Returns:
            z_or_x (Tensor): network output.
            jac (Tensor): log-jacobian-determinant.
        """

        iterator = range(len(self.module_list))
        log_det_jac = 0

        if rev:
            iterator = reversed(iterator)

        if torch.is_tensor(x_or_z):
            x_or_z = (x_or_z,)
        for i in iterator:
            if self.conditions[i] is None:
                x_or_z, j = self.module_list[i](x_or_z, jac=jac, rev=rev)
            else:
                x_or_z, j = self.module_list[i](x_or_z,
                                                c=[c[self.conditions[i]]],
                                                jac=jac, rev=rev)
            log_det_jac = j + log_det_jac

        return x_or_z if self.force_tuple_output else x_or_z[0], log_det_jac
