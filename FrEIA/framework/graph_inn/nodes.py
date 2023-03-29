from collections import deque
from typing import List, Tuple, Iterable, Union, Optional, Type

import torch
from torch import Tensor
from torch.nn import Module

from ...modules.base import InvertibleModule

Connection = Tuple["AbstractNode", int]
ConnectionList = List[Connection]
FlexibleInput = Union["AbstractNode", Connection]
FlexibleInputs = Union[FlexibleInput, Iterable[Connection]]

ModuleType = Union[Type, Module]
InvertibleModuleType = Union[Type, InvertibleModule]


def parse_flexible_inputs(inputs: FlexibleInputs) -> ConnectionList:
    """
    Converts specified inputs to a node to a canonical format.
    Inputs can be specified in three forms:

    - a single node, then this nodes first output is taken as input
    - a single tuple (node, idx), specifying output idx of node
    - a list of tuples [(node, idx)], each specifying output idx of node

    All such formats are converted to the last format.
    """
    if isinstance(inputs, (list, tuple)):
        if len(inputs) == 0:
            return inputs
        if len(inputs) == 2 and isinstance(inputs[1], int):
            return [inputs]
        parsed_inputs = []
        for inp in inputs:
            if isinstance(inp, AbstractNode):
                inp = inp.out0
            elif not (isinstance(inp[0], AbstractNode) and isinstance(inp[1], int)):
                raise ValueError(f"Cannot parse {inp}")
            parsed_inputs.append(inp)
        return parsed_inputs
    else:
        if not isinstance(inputs, AbstractNode):
            raise TypeError(f"Received object of invalid type "
                            f"{type(inputs)} as input.")
        return [(inputs, 0), ]


class AbstractNode:
    def __init__(self, inputs: FlexibleInputs, module_type: InvertibleModuleType, module_args: Optional[dict] = None,
                 conditions=None, name=None):
        if conditions is None:
            conditions = []

        if name:
            self.name = name
        else:
            self.name = hex(id(self))[-6:]
        self.inputs = parse_flexible_inputs(inputs)
        if conditions is not None:
            self.conditions = parse_flexible_inputs(conditions)

        self.module_type = module_type
        self.module_args = module_args

        input_shapes = [input_node.output_dims[node_out_idx]
                        for input_node, node_out_idx in self.inputs]
        condition_shapes = [cond_node.output_dims[cond_node_out_idx]
                            for cond_node, cond_node_out_idx in self.conditions]

        self.input_dims = input_shapes
        self.condition_dims = condition_shapes
        self.module, self.output_dims = self.build_module(condition_shapes, input_shapes)
        self.outputs: List[Optional[Connection]] = [None] * len(self.output_dims)

        # Notify preceding nodes that their output ends up here
        # Entry at position co -> (n, ci) means:
        # My output co goes to input channel ci of n.
        for in_idx, (in_node, out_idx) in enumerate(self.inputs):
            in_node.consume_output(out_idx, self, in_idx)

        # Enable .outX access
        for i in range(len(self.output_dims)):
            self.__dict__[f"out{i}"] = self, i

    def rev_input(self, out_idx):
        """
        Compute which value should be read in reverse mode.
        """
        raise NotImplementedError

    def rev_conditions(self):
        """
        Compute the conditioning for reverse mode -- they move from the
        forward output node of the conditioning node to the reverse output
        this output is fed to.
        """
        return [
            cond_node.rev_input(out_idx)
            for cond_node, out_idx
            in self.conditions
        ]

    def build_module(self, condition_shapes: List[Tuple[int]], input_shapes: List[Tuple[int]]):
        """
        Instantiates the module and determines the output dimension.
        """
        raise NotImplementedError

    def consume_output(self, out_idx: int, by_node: "AbstractNode", in_idx: int):
        if self.outputs[out_idx] is not None:
            raise ValueError(f"Output {out_idx} of node {self} is already consumed.")
        self.outputs[out_idx] = by_node, in_idx

    def forward(self, x_or_z: Iterable[Tensor],
                c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True) \
            -> Tuple[Tuple[Tensor], Union[Tensor, None]]:
        raise NotImplementedError

    def __str__(self):
        module_hint = (self.module_type.__name__ if self.module_type is not None
                       else "")
        name_hint = f" {self.name!r}" if self.name is not None else ""
        return f"{self.__class__.__name__}{name_hint}: {self.input_dims} -> " \
               f"{module_hint} -> {self.output_dims}"

    def __repr__(self):
        name_hint = f" {self.name!r}" if self.name is not None else ""
        return f"{self.__class__.__name__}{name_hint}"


class Node(AbstractNode):
    """
    This Node class represents one transformation in the graph, with an
    arbitrary number of in- and outputs. These in and outputs count as
    being invertible, so they must be consumed.

    The user specifies the input, and the underlying module computes the
    number of outputs.
    """
    def rev_input(self, out_idx):
        # Conditioning on InputNode means conditioning on the node that consumes this input
        return self.outputs[out_idx]

    def build_module(self, condition_shapes: List[Tuple[int]], input_shapes: List[Tuple[int]]) \
            -> Tuple[InvertibleModule, List[Tuple[int]]]:
        """
        Instantiates the module and determines the output dimension by
        calling InvertibleModule#output_dims.
        """
        if isinstance(self.module_type, InvertibleModule):
            if self.module_args is not None:
                raise ValueError(
                    f"You passed an instance of {self.module_type.__class__.__name__}, "
                    f"but also provided argument to its constructor (module_args != None)."
                )
            module = self.module_type
            self.module_type = type(module)
            if module.dims_in != input_shapes:
                raise ValueError(
                    f"You passed an instance of {self.module_type.__class__.__name__} "
                    f"which expects {self.module.dims_in} input shape, "
                    f"but the input shape to the Node is {input_shapes}."
                )
        else:
            module_args = self.module_args
            if module_args is None:
                module_args = {}
            if len(self.conditions) > 0:
                module = self.module_type(input_shapes, dims_c=condition_shapes,
                                          **module_args)
            else:
                module = self.module_type(input_shapes, **module_args)
        return module, module.output_dims(input_shapes)

    def forward(self, x_or_z: Iterable[Tensor],
                c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True) \
            -> Tuple[Tuple[Tensor], Tensor]:
        if len(self.conditions) > 0:
            mod_out = self.module(x_or_z, c=c, rev=rev, jac=jac)
        else:
            mod_out = self.module(x_or_z, rev=rev, jac=jac)

        if torch.is_tensor(mod_out):
            raise ValueError(
                f"The node {self}'s module returned a tensor only. This "
                f"is deprecated without fallback. Please follow the "
                f"signature of InvertibleOperator#forward in your module "
                f"if you want to use it in a GraphINN.")

        if len(mod_out) != 2:
            raise ValueError(
                f"The node {self}'s module returned a tuple of length "
                f"{len(mod_out)}, but should return a tuple `z_or_x, jac`.")

        out, mod_jac = mod_out

        if torch.is_tensor(out):
            raise ValueError(f"The node {self}'s module returns a tensor. "
                             f"This is deprecated.")

        if len(out) != len(self.inputs if rev else self.outputs):
            raise ValueError(
                f"The node {self}'s module returned {len(out)} output "
                f"variables, but should return "
                f"{len(self.inputs if rev else self.outputs)}.")

        if not torch.is_tensor(mod_jac) or mod_jac.shape[0] != out[0].shape[0]:
            if isinstance(mod_jac, (float, int, torch.Tensor)):
                mod_jac = torch.zeros(out[0].shape[0]).to(out[0].device) \
                          + mod_jac
            elif jac:
                raise ValueError(
                    f"The node {self}'s module returned a non-tensor as "
                    f"Jacobian: {mod_jac}")
            elif not jac and mod_jac is not None:
                raise ValueError(
                    f"The node {self}'s module returned neither None nor a "
                    f"Jacobian: {mod_jac}")

        return out, mod_jac


class InputNode(AbstractNode):
    """
    Special type of node that represents the input data of the whole net (or the
    output when running reverse)
    """

    def __init__(self, *dims: int, name=None):
        if len(dims) == 0:
            raise ValueError(f"Input node got empty shape as input.")

        self.dims = dims
        super().__init__([], None, {}, name=name)

    def rev_input(self, out_idx):
        # Conditioning on InputNode means conditioning on the node that consumes this input
        return self.outputs[out_idx]

    def build_module(self, condition_shapes: List[Tuple[int]], input_shapes: List[Tuple[int]]) \
            -> Tuple[None, List[Tuple[int]]]:
        if len(condition_shapes) > 0:
            raise ValueError(f"{self.__class__.__name__} does not accept conditions")
        assert len(input_shapes) == 0, "Forbidden by constructor"
        return None, [self.dims]

    def forward(self, x_or_z: Iterable[Tensor],
                c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True) -> None:
        raise AssertionError(f"{self.__class__.__name__} should never be executed.")


class ConditionNode(AbstractNode):
    """
    Special type of node that represents conditional input to the internal
    networks inside coupling layers.
    """

    def __init__(self, *dims: int, name=None):
        if len(dims) == 0:
            raise ValueError(f"Input node got empty shape as input.")

        self.dims = dims
        super().__init__([], None, {}, name=name)
        # This node does not have consumable outputs
        self.outputs = []

    def rev_input(self, out_idx):
        # No change between forward and backward
        return self, out_idx

    def consume_output(self, out_idx: int, by_node: AbstractNode, in_idx: int):
        raise TypeError(f"{self.__class__.__name__}'s outputs cannot be consumed as "
                        f"inputs to another node, only as conditions.")

    def build_module(self, condition_shapes: List[Tuple[int]], input_shapes: List[Tuple[int]]) \
            -> Tuple[None, List[Tuple[int]]]:
        if len(condition_shapes) > 0:
            raise ValueError(f"{self.__class__.__name__} does not accept conditions")
        assert len(input_shapes) == 0, "Forbidden by constructor"
        return None, [self.dims]

    def forward(self, x_or_z: Iterable[Tensor],
                c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True) -> None:
        raise AssertionError(f"{self.__class__.__name__} should never be executed.")


class OutputNode(AbstractNode):
    """
    Special type of node that represents the output of the whole net (or the
    input when running in reverse).
    """

    def __init__(self, in_node: FlexibleInput, name=None):
        super().__init__(in_node, None, {}, name=name)

    def rev_input(self, out_idx):
        raise ValueError("The output of an output node cannot be accessed. (probably FrEIA bug)")

    def consume_output(self, out_idx: int, by_node: AbstractNode, in_idx: int):
        raise TypeError(f"{self.__class__.__name__}'s outputs cannot be consumed as "
                        f"inputs to another node.")

    def build_module(self, condition_shapes, input_shapes) -> Tuple[None, List[Tuple[int]]]:
        if len(condition_shapes) > 0:
            raise ValueError(f"{self.__class__.__name__} does not accept conditions")
        if len(input_shapes) != 1:
            raise ValueError(f"Output node received {len(input_shapes)} inputs,"
                             f"but only single input is allowed.")
        return None, []

    def forward(self, x_or_z: Iterable[Tensor], c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = True) -> None:
        raise AssertionError(f"{self.__class__.__name__} should never be executed.")


class FeedForwardNode(AbstractNode):
    """
    Special type of node that computes output without Jacobian and inverse.
    """

    def __init__(self, conditions: FlexibleInputs, output_dims: Tuple[int],
                 module_type: ModuleType, module_args: Optional[dict] = None, name=None):
        self.module_output_dims = output_dims
        super().__init__([], module_type, module_args, conditions=conditions, name=name)
        # This node does not have consumable outputs
        self.outputs = []

    def rev_input(self, out_idx):
        # No change between forward and backward
        return self, out_idx

    def build_module(self, condition_shapes: List[Tuple[int]], input_shapes: List[Tuple[int]]) \
            -> Tuple[Module, List[Tuple[int]]]:
        """
        Instantiates the module and determines the output dimension by
        calling InvertibleModule#output_dims.
        """
        if isinstance(self.module_type, Module):
            if self.module_args is not None:
                raise ValueError(
                    f"You passed an instance of {self.module_type.__class__.__name__}, "
                    f"but also provided argument to its constructor (module_args != None)."
                )
            module = self.module_type
            self.module_type = type(module)
        else:
            module_args = self.module_args
            if module_args is None:
                module_args = {}
            module = self.module_type(**module_args)
        return module, [self.module_output_dims]

    def forward(self, x_or_z: Iterable[Tensor],
                c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True) -> Tuple[Tuple[Tensor], None]:
        if isinstance(self.module, InvertibleModule):
            return self.module(c)
        return (self.module(*c),), None


def collect_nodes(*input_nodes: AbstractNode):
    nodes = []
    pending_nodes = deque(input_nodes)

    while len(pending_nodes) > 0:
        start_node = pending_nodes.popleft()
        nodes.append(start_node)
        for node, _ in start_node.outputs + start_node.conditions:
            if node not in nodes and node not in pending_nodes:
                pending_nodes.append(node)

    return nodes
