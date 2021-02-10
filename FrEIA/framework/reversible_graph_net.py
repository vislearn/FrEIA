import warnings
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Iterable, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..modules.base import InvertibleModule


class Node:
    """
    The Node class represents one transformation in the graph, with an
    arbitrary number of in- and outputs.

    The user specifies the input, and the underlying module computes the
    number of outputs.
    """
    module: InvertibleModule

    input_dims: List[Tuple[int]]
    condition_shapes: List[Tuple[int]]
    output_dims: List[Tuple[int]]

    def __init__(self, inputs: Union["Node", Tuple["Node", int], Iterable[Tuple["Node", int]]], module_type, module_args: dict, conditions=None, name=None):
        if conditions is None:
            conditions = []

        if name:
            self.name = name
        else:
            self.name = hex(id(self))[-6:]
        for i in range(256):
            exec('self.out{0} = (self, {0})'.format(i))

        self.inputs = self.parse_inputs(inputs)
        if isinstance(conditions, (list, tuple)):
            self.conditions = conditions
        else:
            self.conditions = [conditions, ]

        # Entry at position co -> (n, ci) means: My output co goes to input channel ci of n.
        self.outputs: List[Tuple[Node, int]] = []
        self.module_type = module_type
        self.module_args = module_args

    def parse_inputs(self, inputs: Union["Node", Tuple["Node", int], Iterable[Tuple["Node", int]]]) -> List[Tuple["Node", int]]:
        """
        Converts specified inputs to a node to a canonical format.
        Inputs can be specified in three forms:

        - a single node, then this nodes first output is taken as input
        - a single tuple (node, idx), specifying output idx of node
        - a list of tuples [(node, idx)], each specifying output idx of node

        All such formats are converted to the last format.
        """
        if isinstance(inputs, (list, tuple)):
            if isinstance(inputs[0], (list, tuple)):
                pass
            elif len(inputs) == 2:
                return [inputs, ]
            else:
                raise RuntimeError(f"Cannot parse inputs provided to node '{self.name}'.")
        else:
            assert isinstance(inputs, Node), "Received object of invalid type " \
                                             f"({type(inputs)}) as input for node '{self.name}'."
            inputs = [(inputs, 0), ]

        for in_idx, (in_node, out_idx) in enumerate(inputs):
            in_node.outputs.append((self, in_idx))

        return inputs

    def build_module(self, input_shapes, condition_shapes, verbose=True):
        """
        Initialize the pytorch nn.Module of this node and determine the output shapes.
        """
        assert len(input_shapes) == len(self.inputs)
        assert len(condition_shapes) == len(self.conditions)

        assert self.module is None, f"Node {self} has been initialised before. This may occur if you insert a node several times into a ReversibleGraphNet."

        self.input_dims = input_shapes
        self.condition_shapes = condition_shapes

        if len(self.conditions) > 0:
            self.module = self.module_type(input_shapes, dims_c=condition_shapes, **self.module_args)
        else:
            self.module = self.module_type(self.input_dims, **self.module_args)

        if verbose:
            print(f"Node '{self.name}' takes the following inputs:")
            for d, (n, c) in zip(self.input_dims, self.inputs):
                print(f"\t Output #{c} of node '{n.name}' with dims {d}")
            for c in self.conditions:
                print(f"\t conditioned on node '{c.name}' with dims {c.data.shape}")
            print()

        self.output_dims = self.module.output_dims(input_shapes)


class InputNode(Node):
    """
    Special type of node that represents the input data of the whole net (or the
    output when running reverse)
    """

    def __init__(self, *dims: int, name=None):
        super().__init__([], None, {}, name=name)
        self.output_dims = [dims]

    def build_module(self, input_shapes, condition_shapes, verbose=True):
        raise AssertionError("build_module on an InputNode should never be called.")


class ConditionNode(Node):
    """
    Special type of node that represents contitional input to the internal
    networks inside coupling layers.
    """

    def __init__(self, *dims: int, name=None):
        super().__init__([], None, {}, name=name)
        self.output_dims = [dims]

    def build_module(self, input_shapes, condition_shapes, verbose=True):
        raise AssertionError("build_module on an InputNode should never be called.")


class OutputNode(Node):
    """
    Special type of node that represents the output of the whole net (or the
    input when running in reverse)
    """

    def __init__(self, name=None):
        super().__init__([], None, {}, name=name)

    def build_module(self, input_shapes, condition_shapes, verbose=True):
        raise AssertionError("build_module on an InputNode should never be called.")


class ReversibleGraphNet(InvertibleModule):
    """
    This class represents the invertible net itself. It is a subclass of
    InvertibleModule and supports the same methods.

    The forward method has an additional option 'rev', with which the net can be
    computed in reverse. Passing `jac` to the forward method additionally computes
    the log determinant of the (inverse) Jacobian of the forward (backward) pass.
    """

    def __init__(self, node_list, ind_in=None, ind_out=None, verbose=True, force_tuple_output=False):
        # Gather lists of input, output and condition nodes
        if ind_in is not None:
            warnings.warn("Use of 'ind_in' and 'ind_out' for ReversibleGraphNet is deprecated, " +
                          "input and output nodes are detected automatically.")
            if isinstance(ind_in, int):
                ind_in = [ind_in]

            in_nodes = [node_list[i] for i in ind_in]
        else:
            in_nodes = [node_list[i] for i in range(len(node_list))
                        if isinstance(node_list[i], InputNode)]
        assert len(in_nodes) > 0, "No input nodes specified."

        if ind_out is not None:
            warnings.warn("Use of 'ind_in' and 'ind_out' for ReversibleGraphNet is deprecated, " +
                          "input and output nodes are detected automatically.")
            if isinstance(ind_out, int):
                ind_out = [ind_out]

            out_nodes = [node_list[i] for i in ind_out]
        else:
            out_nodes = [node_list[i] for i in range(len(node_list))
                         if isinstance(node_list[i], OutputNode)]
        assert len(out_nodes) > 0, "No output nodes specified."

        condition_nodes = [i for i in range(len(node_list)) if isinstance(node_list[i], ConditionNode)]

        # Build the graph and tell nodes about their dimensions so that they can build the modules
        node_list = topological_order(in_nodes, node_list, out_nodes)
        node_out_shapes: Dict[Tuple[Node, int], List[Tuple[int]]] = {}
        global_in_shapes: List[Tuple[int]] = []
        for node in self.node_list:
            input_shapes = [node_out_shapes[in_tuple] for in_tuple in node.inputs]
            condition_shapes = [node_out_shapes[con_tuple] for con_tuple in node.conditions]
            node.build_module(input_shapes, condition_shapes, verbose=verbose)
            for out_idx, output_shape in enumerate(node.output_dims):
                node_out_shapes[node, out_idx] = output_shape
            if isinstance(node, InputNode):
                global_in_shapes.extend(node.output_dims)

        # Only now we
        super().__init__(global_in_shapes, )

        # Now we can store everything -- before calling super constructor, nn.Module doesn't allow assigning anything
        self.in_nodes = in_nodes
        self.condition_nodes = condition_nodes
        self.out_nodes = out_nodes

        self.force_tuple_output = force_tuple_output
        self.module_list = nn.ModuleList([n.module for n in node_list])

    def forward(self, x_or_z: Union[Tensor, Iterable[Tensor]], c: Iterable[Tensor], rev: bool = False, jac: bool = True, intermediate_outputs: bool = False) -> Tuple[Tuple[Tensor], Tensor]:
        """
        Forward or backward computation of the whole net.
        """
        jacobian = None
        outs = {}
        for tensor, start_node in zip(x_or_z, self.out_nodes if rev else self.in_nodes):
            outs[start_node, 0] = tensor
        for tensor, condition_node in zip(c, self.condition_nodes):
            outs[condition_node, 0] = tensor

        # Go backwards through nodes if rev=True
        for node in self.node_list[::-1 if rev else 1]:
            has_condition = len(node.conditions) > 0

            mod_in = []
            mod_c = []
            for prev_node, channel in (node.outputs if rev else node.inputs):
                mod_in.append(outs[prev_node, channel])
            for cond_node in node.conditions:
                mod_c.append(outs[cond_node, 0])
            mod_in = tuple(mod_in)
            mod_c = tuple(mod_c)

            if has_condition:
                mod_out = node.module(mod_in, c=mod_c, rev=rev, jac=jac)
            else:
                mod_out = node.module(mod_in, rev=rev, jac=jac)

            if torch.is_tensor(mod_out):
                raise ValueError(f"The node {node}'s module returned a tensor only. This is deprecated without fallback. Please follow the signature of InvertibleOperator#forward in your module if you want to use it in a ReversibleGraphNet.")
            if len(mod_out) != 2:
                raise ValueError(f"The node {node}'s module returned a tuple of length {len(mod_out)}, but should return a tuple `z_or_x, jac`.")

            out, mod_jac = mod_out

            if torch.is_tensor(out):
                # Not according to specification!
                add_text = " Consider passing force_tuple_output=True to the contained ReversibleGraphNet" if isinstance(node.module, ReversibleGraphNet) else ""
                raise ValueError(f"The node {node}'s module returns a tensor.{add_text}")
            if len(out) != len(node.inputs if rev else node.outputs):
                raise ValueError(f"The node {node}'s module returned {len(out)} output variables, but should return {len(node.inputs if rev else node.outputs)}.")
            if not torch.is_tensor(mod_jac):
                if jac:
                    raise ValueError(f"The node {node}'s module returned a non-tensor as Jacobian.")
                elif not jac and mod_jac is not None:
                    raise ValueError(f"The node {node}'s module returned neither None nor a Jacobian.")

            for out_idx, out_value in enumerate(out):
                outs[self, out_idx] = out_value

            if jac:
                jacobian = jacobian + mod_jac

        if intermediate_outputs:
            return outs, jacobian
        else:
            out_list = [outs[(out_node, 0)] for out_node in self.out_nodes]
            if len(out_list) == 1 and not self.force_tuple_output:
                return out_list[0], jacobian
            else:
                return tuple(out_list), jacobian

    def log_jacobian_numerical(self, x, c=None, rev=False, h=1e-04):
        """
        Approximate log Jacobian determinant via finite differences.
        """
        if isinstance(x, (list, tuple)):
            batch_size = x[0].shape[0]
            ndim_x_separate = [np.prod(x_i.shape[1:]) for x_i in x]
            ndim_x_total = sum(ndim_x_separate)
            x_flat = torch.cat([x_i.view(batch_size, -1) for x_i in x], dim=1)
        else:
            batch_size = x.shape[0]
            ndim_x_total = np.prod(x.shape[1:])
            x_flat = x.reshape(batch_size, -1)

        J_num = torch.zeros(batch_size, ndim_x_total, ndim_x_total)
        for i in range(ndim_x_total):
            offset = x[0].new_zeros(batch_size, ndim_x_total)
            offset[:, i] = h
            if isinstance(x, (list, tuple)):
                x_upper = torch.split(x_flat + offset, ndim_x_separate, dim=1)
                x_upper = [x_upper[i].view(*x[i].shape) for i in range(len(x))]
                x_lower = torch.split(x_flat - offset, ndim_x_separate, dim=1)
                x_lower = [x_lower[i].view(*x[i].shape) for i in range(len(x))]
            else:
                x_upper = (x_flat + offset).view(*x.shape)
                x_lower = (x_flat - offset).view(*x.shape)
            y_upper = self.forward(x_upper, c=c, rev=rev, jac=False)
            y_lower = self.forward(x_lower, c=c, rev=rev, jac=False)
            if isinstance(y_upper, (list, tuple)):
                y_upper = torch.cat([y_i.view(batch_size, -1) for y_i in y_upper], dim=1)
                y_lower = torch.cat([y_i.view(batch_size, -1) for y_i in y_lower], dim=1)
            J_num[:, :, i] = (y_upper - y_lower).view(batch_size, -1) / (2 * h)
        logdet_num = x[0].new_zeros(batch_size)
        for i in range(batch_size):
            logdet_num[i] = torch.det(J_num[i, :, :]).abs().log()

        return logdet_num

    def get_node_by_name(self, name) -> Optional[Node]:
        """
        Return the first node in the graph with the provided name.
        """
        for node in self.node_list:
            if node.name == name:
                return node
        return None

    def get_module_by_name(self, name) -> Optional[nn.Module]:
        """
        Return module of the first node in the graph with the provided name.
        """
        node = self.get_node_by_name(name)
        try:
            return node.module
        except AttributeError:
            return None


def topological_order(all_nodes: List[Node], in_nodes: List[InputNode], out_nodes: List[OutputNode]) -> List[Node]:
    """
    Computes the topological order of nodes.

    Parameters:
        all_nodes: All nodes in the computation graph.
        in_nodes: Input nodes (must also be present in `all_nodes`)
        out_nodes: Output nodes (must also be present in `all_nodes`)

    Returns:
        A sorted list of nodes, where the inputs to some node in the list
        are available when all previous nodes in the list have been executed.
    """
    # Edge dicts in both directions
    edges_out_to_in = {node_b: {node_a for node_a, out_idx in node_b.inputs} for node_b in all_nodes + out_nodes}
    edges_in_to_out = defaultdict(list)
    for node_out, node_ins in edges_out_to_in:
        for node_in in node_ins:
            edges_in_to_out[node_in].append(node_out)

    # Kahn's algorithm starting from the output nodes
    sorted_nodes = []
    no_pending_edges = deque(out_nodes)

    while len(no_pending_edges) > 0:
        node = no_pending_edges.popleft()
        sorted_nodes.append(node)
        for in_node in edges_out_to_in[node]:
            edges_out_to_in[node].remove(in_node)
            edges_in_to_out[in_node].remove(node)

            if len(edges_in_to_out) == 0:
                no_pending_edges.append(node)

    for in_node in in_nodes:
        assert in_node in sorted_nodes, f"Error in graph: Input node {in_node} is not connected to any output."

    if sum(map(len, edges_in_to_out)) == 0:
        return sorted_nodes[::-1]
    else:
        raise ValueError("Graph is cyclic.")
