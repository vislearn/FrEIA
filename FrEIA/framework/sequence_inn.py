from typing import Iterable, Tuple, List, Union

import torch.nn as nn
import torch
from torch import Tensor

from FrEIA.modules import InvertibleModule
from FrEIA.modules.inverse import Inverse


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

        if isinstance(module_class, InvertibleModule):
            module = module_class
            if module.dims_in != dims_in:
                raise ValueError(
                    f"You passed an instance of {module.__class__.__name__} to "
                    f"SequenceINN which expects a {module.dims_in} input, "
                    f"but the output of the previous layer is of shape "
                    f"{dims_in}."
                )
            if len(kwargs) > 0:
                raise ValueError(
                    "You try to append an instanciated "
                    "InvertibleModule to SequenceINN, but also provided "
                    "constructor kwargs."
                )
        else:
            if cond is not None:
                kwargs['dims_c'] = [cond_shape]
            module = module_class(dims_in, **kwargs)
        self.module_list.append(module)
        output_dims = module.output_dims(dims_in)
        if len(output_dims) != 1:
            raise ValueError(
                f"Module of type {module.__class__} has more than one output: "
                f"{output_dims}"
            )
        self.shapes.append(output_dims[0])

    def __setitem__(self, key, value: InvertibleModule):
        """
        Replaces the module at position key with value.
        """
        if isinstance(key, slice):
            raise NotImplementedError("Setting sequence_inn[...] with slices as index is not supported.")
        existing_module = self.module_list[key]
        assert isinstance(existing_module, InvertibleModule)

        # Input dims
        if existing_module.dims_in != value.dims_in:
            raise ValueError(
                f"Module at position {key} must have input shape {existing_module.dims_in}, "
                f"but the replacement has input shape {value.dims_in}."
            )

        # Output dims
        existing_dims_out = existing_module.output_dims(existing_module.dims_in)
        target_dims_out = value.output_dims(value.dims_in)
        if existing_dims_out != target_dims_out:
            raise ValueError(
                f"Module at position {key} must have input shape {existing_dims_out}, "
                f"but the replacement has input shape {target_dims_out}."
            )

        # Condition
        if existing_module.dims_c != value.dims_c:
            raise ValueError(
                f"Module at position {key} must have condition shape {existing_dims_out}, "
                f"but the replacement has condition shape {target_dims_out}."
            )

        # Actually replace
        self.module_list[key] = value

    def __getitem__(self, item) -> Union[InvertibleModule, "SequenceINN"]:
        if isinstance(item, slice):
            # Zero-length
            in_dims = self.shapes[item]
            start, stop, stride = item.indices(len(self))
            sub_inn = SequenceINN(*self.shapes[start], force_tuple_output=self.force_tuple_output)
            if len(in_dims) == 0:
                return sub_inn
            cond_map = {None: None}
            cond_counter = 0
            for idx in range(start, stop, stride):
                module = self.module_list[idx]
                module_condition = self.conditions[idx]
                if stride < 0:
                    module = Inverse(module)
                if module_condition not in cond_map:
                    cond_map[module_condition] = cond_counter
                    cond_counter += 1
                sub_inn.append(module, cond_map[module_condition])
            return sub_inn

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
        log_det_jac = torch.zeros(x_or_z.shape[0], device=x_or_z.device)

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
