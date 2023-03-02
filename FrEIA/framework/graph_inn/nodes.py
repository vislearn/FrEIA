import warnings
from collections import deque, defaultdict
from typing import List, Tuple, Iterable, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ...modules.base import InvertibleModule


class Node:
    """
    The Node class represents one transformation in the graph, with an
    arbitrary number of in- and outputs.

    The user specifies the input, and the underlying module computes the
    number of outputs.
    """

    def __init__(self, inputs: Union["Node", Tuple["Node", int],
                                     Iterable[Tuple["Node", int]]],
                 module_type, module_args: Optional[dict] = None, conditions=None, name=None):
        if conditions is None:
            conditions = []

        if name:
            self.name = name
        else:
            self.name = hex(id(self))[-6:]
        self.inputs = self.parse_inputs(inputs)
        if isinstance(conditions, (list, tuple)):
            self.conditions = conditions
        else:
            self.conditions = [conditions, ]

        self.outputs: List[Tuple[Node, int]] = []
        self.module_type = module_type
        self.module_args = module_args

        input_shapes = [input_node.output_dims[node_out_idx]
                        for input_node, node_out_idx in self.inputs]
        condition_shapes = [cond_node.output_dims[0]
                            for cond_node in self.conditions]

        self.input_dims = input_shapes
        self.condition_dims = condition_shapes
        self.module, self.output_dims = self.build_module(condition_shapes,
                                                          input_shapes)

        # Notify preceding nodes that their output ends up here
        # Entry at position co -> (n, ci) means:
        # My output co goes to input channel ci of n.
        for in_idx, (in_node, out_idx) in enumerate(self.inputs):
            in_node.outputs[out_idx] = (self, in_idx)

        # Enable .outX access
        for i in range(len(self.output_dims)):
            self.__dict__[f"out{i}"] = self, i
            self.outputs.append(None)

    def build_module(self, condition_shapes, input_shapes) \
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
        elif len(self.conditions) > 0:
            module = self.module_type(input_shapes, dims_c=condition_shapes,
                                      **self.module_args)
        else:
            module = self.module_type(input_shapes, **self.module_args)
        return module, module.output_dims(input_shapes)

    def parse_inputs(self, inputs: Union["Node", Tuple["Node", int],
                                         Iterable[Tuple["Node", int]]]) \
            -> List[Tuple["Node", int]]:
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
            elif isinstance(inputs[0], (list, tuple)):
                return inputs
            elif len(inputs) == 2:
                return [inputs, ]
            else:
                raise ValueError(
                    f"Cannot parse inputs provided to node '{self.name}'.")
        else:
            if not isinstance(inputs, Node):
                raise TypeError(f"Received object of invalid type "
                                f"({type(inputs)}) as input for node "
                                f"'{self.name}'.")
            return [(inputs, 0), ]

    def __str__(self):
        module_hint = (self.module_type.__name__ if self.module_type is not None
                       else "")
        name_hint = f" {self.name!r}" if self.name is not None else ""
        return f"{self.__class__.__name__}{name_hint}: {self.input_dims} -> " \
               f"{module_hint} -> {self.output_dims}"

    def __repr__(self):
        name_hint = f" {self.name!r}" if self.name is not None else ""
        return f"{self.__class__.__name__}{name_hint}"


class InputNode(Node):
    """
    Special type of node that represents the input data of the whole net (or the
    output when running reverse)
    """

    def __init__(self, *dims: int, name=None):
        self.dims = dims
        super().__init__([], None, {}, name=name)

    def build_module(self, condition_shapes, input_shapes) \
            -> Tuple[None, List[Tuple[int]]]:
        if len(condition_shapes) > 0:
            raise ValueError(
                f"{self.__class__.__name__} does not accept conditions")
        assert len(input_shapes) == 0, "Forbidden by constructor"
        return None, [self.dims]


class ConditionNode(Node):
    """
    Special type of node that represents contitional input to the internal
    networks inside coupling layers.
    """

    def __init__(self, *dims: int, name=None):
        self.dims = dims
        super().__init__([], None, {}, name=name)
        self.outputs: List[Tuple[Node, int]] = []

    def build_module(self, condition_shapes, input_shapes) \
            -> Tuple[None, List[Tuple[int]]]:
        if len(condition_shapes) > 0:
            raise ValueError(
                f"{self.__class__.__name__} does not accept conditions")
        assert len(input_shapes) == 0, "Forbidden by constructor"
        return None, [self.dims]


class OutputNode(Node):
    """
    Special type of node that represents the output of the whole net (or the
    input when running in reverse).
    """

    def __init__(self, in_node: Union[Node, Tuple[Node, int]], name=None):
        super().__init__(in_node, None, {}, name=name)

    def build_module(self, condition_shapes, input_shapes) \
            -> Tuple[None, List[Tuple[int]]]:
        if len(condition_shapes) > 0:
            raise ValueError(
                f"{self.__class__.__name__} does not accept conditions")
        if len(input_shapes) != 1:
            raise ValueError(f"Output node received {len(input_shapes)} inputs,"
                             f"but only single input is allowed.")
        return None, []


