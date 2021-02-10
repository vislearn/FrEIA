import sys
import warnings
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from . import dummy_modules as dummys
from ..modules.base import InvertibleModule


class Node:
    '''The Node class represents one transformation in the graph, with an
    arbitrary number of in- and outputs.'''

    def __init__(self, inputs, module_type, module_args, conditions=[], name=None):
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

        self.outputs = []
        self.module_type = module_type
        self.module_args = module_args

        self.input_dims = None
        self.module = None
        self.computed = None
        self.computed_rev = None
        self.id = None

    def parse_inputs(self, inputs):
        if isinstance(inputs, (list, tuple)):
            if isinstance(inputs[0], (list, tuple)):
                return inputs
            elif len(inputs) == 2:
                return [inputs, ]
            else:
                raise RuntimeError(f"Cannot parse inputs provided to node '{self.name}'.")
        else:
            assert isinstance(inputs, Node), "Received object of invalid type " \
                                             f"({type(inputs)}) as input for node '{self.name}'."
            return [(inputs, 0), ]

    def build_modules(self, verbose=True):
        ''' Returns a list with the dimension of each output of this node,
        recursively calling build_modules of the nodes connected to the input.
        Use this information to initialize the pytorch nn.Module of this node.
        '''

        if not self.input_dims:  # Only do it if this hasn't been computed yet
            self.input_dims = [n.build_modules(verbose=verbose)[c]
                               for n, c in self.inputs]
            try:
                if len(self.conditions) > 0:
                    c_dims = [c.build_modules(verbose=verbose)[0] for c in self.conditions]
                    self.module = self.module_type(self.input_dims, dims_c=c_dims,
                                                   **self.module_args)
                else:
                    self.module = self.module_type(self.input_dims,
                                                   **self.module_args)
            except Exception as e:
                print('Error in node %s' % (self.name))
                raise e

            if verbose:
                print(f"Node '{self.name}' takes the following inputs:")
                for d, (n, c) in zip(self.input_dims, self.inputs):
                    print(f"\t Output #{c} of node '{n.name}' with dims {d}")
                for c in self.conditions:
                    print(f"\t conditioned on node '{c.name}' " +
                          f"with dims {c.data.shape}")
                print()

            self.output_dims = self.module.output_dims(self.input_dims)
            self.n_outputs = len(self.output_dims)

        return self.output_dims

    def run_forward(self, op_list):
        '''Determine the order of operations needed to reach this node. Calls
        run_forward of parent nodes recursively. Each operation is appended to
        the global list op_list, in the form (node ID, input variable IDs,
        output variable IDs)'''

        if not self.computed:

            # Compute all nodes which provide inputs, filter out the
            # channels you need
            self.input_vars = []
            for i, (n, c) in enumerate(self.inputs):
                self.input_vars.append(n.run_forward(op_list)[c])
                # Register self as an output in the input node
                n.outputs.append((self, i))
            # Compute all nodes which provide conditioning
            self.condition_vars = []
            for i, c in enumerate(self.conditions):
                self.condition_vars.append(c.run_forward(op_list)[0])
                # Register self as an output in the condition node
                c.outputs.append((self, i))

            # All outputs could now be computed
            self.computed = [(self.id, i) for i in range(self.n_outputs)]
            op_list.append((self.id, self.input_vars, self.computed, self.condition_vars))

        # Return the variables you have computed (this happens mulitple times
        # without recomputing if called repeatedly)
        return self.computed

    def run_backward(self, op_list):
        '''See run_forward, this is the same, only for the reverse computation.
        Need to call run_forward first, otherwise this function will not
        work'''

        assert len(self.outputs) > 0, "Call run_forward first"
        if not self.computed_rev:

            # These are the input variables that must be computed first
            output_vars = [(self.id, i) for i in range(self.n_outputs)]

            # Recursively compute these
            for n, c in self.outputs:
                n.run_backward(op_list)

            # The variables that this node computes are the input variables
            # from the forward pass
            self.computed_rev = self.input_vars
            if len(self.condition_vars) == 0:
                self.condition_vars = [c.run_forward(op_list)[0] for c in self.conditions]
            op_list.append((self.id, output_vars, self.computed_rev, self.condition_vars))

        return self.computed_rev


class InputNode(Node):
    '''Special type of node that represents the input data of the whole net (or
    ouput when running reverse)'''

    def __init__(self, *dims, name='node'):
        self.name = name
        self.data = dummys.dummy_data(*dims)
        self.outputs = []
        self.conditions = []
        self.condition_vars = []
        self.module = None
        self.computed_rev = None
        self.n_outputs = 1
        self.input_vars = []
        self.out0 = (self, 0)

    def build_modules(self, verbose=True):
        return [self.data.shape]

    def run_forward(self, op_list):
        return [(self.id, 0)]


class ConditionNode(Node):
    '''Special type of node that represents contitional input to the internal
    networks inside coupling layers'''

    def __init__(self, *dims, name='node'):
        self.name = name
        self.data = dummys.dummy_data(*dims)
        self.outputs = []
        self.conditions = []
        self.condition_vars = []
        self.module = None
        self.computed_rev = None
        self.n_outputs = 1
        self.input_vars = []
        self.out0 = (self, 0)

    def build_modules(self, verbose=True):
        return [self.data.shape]

    def run_forward(self, op_list):
        return [(self.id, 0)]


class OutputNode(Node):
    '''Special type of node that represents the output of the whole net (of the
    input when running in reverse)'''

    class dummy(nn.Module):

        def __init__(self, *args):
            super().__init__()

        def __call__(*args):
            return args

        def output_dims(*args):
            return args

    def __init__(self, inputs, name='node'):
        self.module_type, self.module_args = self.dummy, {}
        self.output_dims = []
        self.inputs = self.parse_inputs(inputs)
        self.conditions = []
        self.input_dims, self.module = None, None
        self.computed = None
        self.id = None
        self.name = name

        for c, inp in enumerate(self.inputs):
            inp[0].outputs.append((self, c))

    def run_backward(self, op_list):
        return [(self.id, 0)]


class ReversibleGraphNet(InvertibleModule):
    '''This class represents the invertible net itself. It is a subclass of
    torch.nn.Module and supports the same methods. The forward method has an
    additional option 'rev', whith which the net can be computed in reverse.'''

    def __init__(self, node_list, ind_in=None, ind_out=None, verbose=True):
        """
        todo properly inherit signature
        """
        super().__init__()

        # Gather lists of input, output and condition nodes
        if ind_in is not None:
            warnings.warn("Use of 'ind_in' and 'ind_out' for ReversibleGraphNet is deprecated, " +
                          "input and output nodes are detected automatically.")
            if isinstance(ind_in, int):
                ind_in = [ind_in]

            self.in_nodes = [node_list[i] for i in ind_in]
        else:
            self.in_nodes = [node_list[i] for i in range(len(node_list))
                             if isinstance(node_list[i], InputNode)]
        assert len(self.in_nodes) > 0, "No input nodes specified."

        if ind_out is not None:
            warnings.warn("Use of 'ind_in' and 'ind_out' for ReversibleGraphNet is deprecated, " +
                          "input and output nodes are detected automatically.")
            if isinstance(ind_out, int):
                self.ind_out = [ind_out]

            self.out_nodes = [node_list[i] for i in ind_in]
        else:
            self.out_nodes = [node_list[i] for i in range(len(node_list))
                              if isinstance(node_list[i], OutputNode)]
        assert len(self.out_nodes) > 0, "No output nodes specified."

        self.condition_nodes = [i for i in range(len(node_list))
                                if isinstance(node_list[i], ConditionNode)]

        # Build the graph and tell nodes about their dimensions so that they can build the modules
        self.node_list = topological_order(self.in_nodes, node_list, self.out_nodes)
        node_out_shapes: Dict[Tuple[Node, int], Iterable[Tuple[int]]] = {}
        for node in self.node_list:
            input_shapes = [node_out_shapes[in_tuple] for in_tuple in node.inputs]
            condition_shapes = [node_out_shapes[con_tuple] for con_tuple in node.conditions]
            node.build_modules(input_shapes, condition_shapes, verbose=verbose)
            for out_idx, output_shape in enumerate(node.output_dims):
                node_out_shapes[node, out_idx] = output_shape
            # todo input shapes

        self.module_list = nn.ModuleList([n.module for n in node_list])

    def forward(self, x_or_z: Iterable[Tensor], c: Iterable[Tensor], rev: bool = False, jac: bool = True, intermediate_outputs: bool=False) -> Tuple[Tuple[Tensor], Tensor]:
        """
        Forward or backward computation of the whole net.
        """
        # Go backwards throguh nodes if rev=True
        jacobian = None
        # todo fill with input/output values
        outs = {}
        for node in self.node_list[::1 if not rev else -1]:
            has_condition = len(node.conditions) > 0
            if has_condition:
                out, mod_jac = node.module(x_or_z, c=c, rev=rev, jac=jac)
            else:
                out, mod_jac = node.module(x_or_z, rev=rev, jac=jac)
            if jac:
                jacobian = jacobian + mod_jac

        if intermediate_outputs:
            return outs
        else:
            return [outs[(out_node, 0)] for out_node in self.out_nodes]

    def log_jacobian_numerical(self, x, c=None, rev=False, h=1e-04):
        '''Approximate log Jacobian determinant via finite differences.'''
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
            y_upper = self.forward(x_upper, c=c)
            y_lower = self.forward(x_lower, c=c)
            if isinstance(y_upper, (list, tuple)):
                y_upper = torch.cat([y_i.view(batch_size, -1) for y_i in y_upper], dim=1)
                y_lower = torch.cat([y_i.view(batch_size, -1) for y_i in y_lower], dim=1)
            J_num[:, :, i] = (y_upper - y_lower).view(batch_size, -1) / (2 * h)
        logdet_num = x[0].new_zeros(batch_size)
        for i in range(batch_size):
            logdet_num[i] = torch.det(J_num[i, :, :]).abs().log()

        return logdet_num

    def load_state_dict(self, state_dict, *args, **kwargs):

        state_dict_no_buffers = {}
        for k, p in state_dict.items():
            if k in self._buffers and self._buffers[k] is None:
                continue
            state_dict_no_buffers[k] = p

        return super().load_state_dict(state_dict_no_buffers, *args, **kwargs)

    def get_node_by_name(self, name):
        # Return the first node in the graph with the provided name
        for node in self.node_list:
            if node.name == name:
                return node
        return None

    def get_module_by_name(self, name):
        # Return module of the first node in the graph with the provided name
        node = self.get_node_by_name(name)
        try:
            return node.module
        except:
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
