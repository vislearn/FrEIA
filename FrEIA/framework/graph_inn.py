import warnings
from collections import deque, defaultdict
from typing import List, Tuple, Iterable, Union, Optional

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


class GraphINN(InvertibleModule):
    """
    This class represents the invertible net itself. It is a subclass of
    InvertibleModule and supports the same methods.

    The forward method has an additional option 'rev', with which the net can be
    computed in reverse. Passing `jac` to the forward method additionally
    computes the log determinant of the (inverse) Jacobian of the forward
    (backward) pass.
    """

    def __init__(self, node_list, force_tuple_output=False, verbose=False):
        # Gather lists of input, output and condition nodes
        in_nodes = [node_list[i] for i in range(len(node_list))
                    if isinstance(node_list[i], InputNode)]
        out_nodes = [node_list[i] for i in range(len(node_list))
                     if isinstance(node_list[i], OutputNode)]
        condition_nodes = [node_list[i] for i in range(len(node_list)) if
                           isinstance(node_list[i], ConditionNode)]

        # Check that all nodes are in the list
        for node in node_list:
            for in_node, idx in node.inputs:
                if in_node not in node_list:
                    raise ValueError(f"{node} gets input from {in_node}, "
                                     f"but the latter is not in the node_list "
                                     f"passed to GraphINN.")
            for out_node, idx in node.outputs:
                if out_node not in node_list:
                    raise ValueError(f"{out_node} gets input from {node}, "
                                     f"but the it's not in the node_list "
                                     f"passed to GraphINN.")

        # Build the graph and tell nodes about their dimensions so that they can
        # build the modules
        node_list = topological_order(node_list, in_nodes, out_nodes)
        global_in_shapes = [node.output_dims[0] for node in in_nodes]
        global_out_shapes = [node.input_dims[0] for node in out_nodes]
        global_cond_shapes = [node.output_dims[0] for node in condition_nodes]

        # Only now we can set out shapes
        super().__init__(global_in_shapes, global_cond_shapes)
        self.node_list = node_list

        # Now we can store everything -- before calling super constructor,
        # nn.Module doesn't allow assigning anything
        self.in_nodes = in_nodes
        self.condition_nodes = condition_nodes
        self.out_nodes = out_nodes

        self.global_out_shapes = global_out_shapes
        self.force_tuple_output = force_tuple_output
        self.module_list = nn.ModuleList([n.module for n in node_list
                                          if n.module is not None])

        if verbose:
            print(self)

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        if len(self.global_out_shapes) == 1 and not self.force_tuple_output:
            raise ValueError("You can only call output_dims on a "
                             "GraphINN with more than one output "
                             "or when setting force_tuple_output=True.")
        return self.global_out_shapes

    def forward(self, x_or_z: Union[Tensor, Iterable[Tensor]],
                c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True,
                intermediate_outputs: bool = False, x: None = None) \
            -> Tuple[Tuple[Tensor], Tensor]:
        """
        Forward or backward computation of the whole net.
        """
        if x is not None:
            x_or_z = x
            warnings.warn("You called GraphINN(x=...). x is now called x_or_z, "
                          "please pass input as positional argument.")

        if torch.is_tensor(x_or_z):
            x_or_z = x_or_z,
        if torch.is_tensor(c):
            c = c,

        jacobian = torch.zeros(x_or_z[0].shape[0]).to(x_or_z[0])
        outs = {}
        jacobian_dict = {} if jac else None

        # Explicitly set conditions and starts
        start_nodes = self.out_nodes if rev else self.in_nodes
        if len(x_or_z) != len(start_nodes):
            raise ValueError(f"Got {len(x_or_z)} inputs, but expected "
                             f"{len(start_nodes)}.")
        for tensor, start_node in zip(x_or_z, start_nodes):
            outs[start_node, 0] = tensor

        if c is None:
            c = []
        if len(c) != len(self.condition_nodes):
            raise ValueError(f"Got {len(c)} conditions, but expected "
                             f"{len(self.condition_nodes)}.")
        for tensor, condition_node in zip(c, self.condition_nodes):
            outs[condition_node, 0] = tensor

        # Go backwards through nodes if rev=True
        for node in self.node_list[::-1 if rev else 1]:
            # Skip all special nodes
            if node in self.in_nodes + self.out_nodes + self.condition_nodes:
                continue

            has_condition = len(node.conditions) > 0

            mod_in = []
            mod_c = []
            for prev_node, channel in (node.outputs if rev else node.inputs):
                mod_in.append(outs[prev_node, channel])
            for cond_node in node.conditions:
                mod_c.append(outs[cond_node, 0])
            mod_in = tuple(mod_in)
            mod_c = tuple(mod_c)

            try:
                if has_condition:
                    mod_out = node.module(mod_in, c=mod_c, rev=rev, jac=jac)
                else:
                    mod_out = node.module(mod_in, rev=rev, jac=jac)
            except Exception as e:
                raise RuntimeError(f"{node} encountered an error.") from e

            out, mod_jac = self._check_output(node, mod_out, jac, rev)

            for out_idx, out_value in enumerate(out):
                outs[node, out_idx] = out_value

            if jac:
                jacobian = jacobian + mod_jac
                jacobian_dict[node] = mod_jac

        for out_node in (self.in_nodes if rev else self.out_nodes):
            # This copies the one input of the out node
            outs[out_node, 0] = outs[(out_node.outputs if rev
                                      else out_node.inputs)[0]]

        if intermediate_outputs:
            return outs, jacobian_dict
        else:
            out_list = [outs[out_node, 0] for out_node
                        in (self.in_nodes if rev else self.out_nodes)]
            if len(out_list) == 1 and not self.force_tuple_output:
                return out_list[0], jacobian
            else:
                return tuple(out_list), jacobian

    def _check_output(self, node, mod_out, jac, rev):
        if torch.is_tensor(mod_out):
            raise ValueError(
                f"The node {node}'s module returned a tensor only. This "
                f"is deprecated without fallback. Please follow the "
                f"signature of InvertibleOperator#forward in your module "
                f"if you want to use it in a GraphINN.")

        if len(mod_out) != 2:
            raise ValueError(
                f"The node {node}'s module returned a tuple of length "
                f"{len(mod_out)}, but should return a tuple `z_or_x, jac`.")

        out, mod_jac = mod_out

        if torch.is_tensor(out):
            raise ValueError(f"The node {node}'s module returns a tensor. "
                             f"This is deprecated.")

        if len(out) != len(node.inputs if rev else node.outputs):
            raise ValueError(
                f"The node {node}'s module returned {len(out)} output "
                f"variables, but should return "
                f"{len(node.inputs if rev else node.outputs)}.")

        if not torch.is_tensor(mod_jac):
            if isinstance(mod_jac, (float, int)):
                mod_jac = torch.zeros(out[0].shape[0]).to(out[0].device) \
                          + mod_jac
            elif jac:
                raise ValueError(
                    f"The node {node}'s module returned a non-tensor as "
                    f"Jacobian: {mod_jac}")
            elif not jac and mod_jac is not None:
                raise ValueError(
                    f"The node {node}'s module returned neither None nor a "
                    f"Jacobian: {mod_jac}")
        return out, mod_jac

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
            y_upper, _ = self.forward(x_upper, c=c, rev=rev, jac=False)
            y_lower, _ = self.forward(x_lower, c=c, rev=rev, jac=False)
            if isinstance(y_upper, (list, tuple)):
                y_upper = torch.cat(
                    [y_i.view(batch_size, -1) for y_i in y_upper], dim=1)
                y_lower = torch.cat(
                    [y_i.view(batch_size, -1) for y_i in y_lower], dim=1)
            J_num[:, :, i] = (y_upper - y_lower).view(batch_size, -1) / (2 * h)
        logdet_num = x[0].new_zeros(batch_size)
        for i in range(batch_size):
            logdet_num[i] = torch.slogdet(J_num[i])[1]

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


def topological_order(all_nodes: List[Node], in_nodes: List[InputNode],
                      out_nodes: List[OutputNode]) -> List[Node]:
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
    edges_out_to_in = {node_b: {node_a for node_a, out_idx in node_b.inputs} for
                       node_b in all_nodes + out_nodes}
    edges_in_to_out = defaultdict(set)
    for node_out, node_ins in edges_out_to_in.items():
        for node_in in node_ins:
            edges_in_to_out[node_in].add(node_out)

    # Kahn's algorithm starting from the output nodes
    sorted_nodes = []
    no_pending_edges = deque(out_nodes)

    while len(no_pending_edges) > 0:
        node = no_pending_edges.popleft()
        sorted_nodes.append(node)
        for in_node in list(edges_out_to_in[node]):
            edges_out_to_in[node].remove(in_node)
            edges_in_to_out[in_node].remove(node)

            if len(edges_in_to_out[in_node]) == 0:
                no_pending_edges.append(in_node)

    for in_node in in_nodes:
        if in_node not in sorted_nodes:
            raise ValueError(f"Error in graph: {in_node} is not connected "
                             f"to any output.")

    if sum(map(len, edges_in_to_out.values())) == 0:
        return sorted_nodes[::-1]
    else:
        raise ValueError("Graph is cyclic.")
