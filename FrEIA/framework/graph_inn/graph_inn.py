import warnings
from collections import deque, defaultdict
from typing import List, Tuple, Iterable, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ...modules.base import InvertibleModule
from .nodes import AbstractNode, Node, ConditionNode, InputNode, OutputNode, FeedForwardNode


class GraphINN(InvertibleModule):
    """
    This class represents the invertible net itself. It is a subclass of
    InvertibleModule and supports the same methods.

    The forward method has an additional option 'rev', with which the net can be
    computed in reverse. Passing `jac` to the forward method additionally
    computes the log determinant of the (inverse) Jacobian of the forward
    (backward) pass.
    """

    def __init__(self, node_list: Iterable[AbstractNode], force_tuple_output=False, verbose=False):
        # Gather lists of input, output and condition nodes
        in_nodes = [node for node in node_list if isinstance(node, InputNode)]
        out_nodes = [node for node in node_list if isinstance(node, OutputNode)]
        condition_nodes = [node for node in node_list if isinstance(node, ConditionNode)]
        ff_nodes = [node for node in node_list if isinstance(node, FeedForwardNode)]

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
                                     f"but it's not in the node_list "
                                     f"passed to GraphINN.")
            for cond_node, idx in node.conditions:
                if cond_node not in node_list:
                    raise ValueError(f"{node} is conditioned on {cond_node}, "
                                     f"but the latter not in the node_list "
                                     f"passed to GraphINN.")

        # Global in- and output
        global_in_shapes = [node.output_dims[0] for node in in_nodes]
        global_out_shapes = [node.input_dims[0] for node in out_nodes]
        global_cond_shapes = [node.output_dims[0] for node in condition_nodes]

        # Only now we can set out shapes
        super().__init__(global_in_shapes, global_cond_shapes)
        self.node_list_fwd = topological_order(node_list, in_nodes, out_nodes, rev=False)
        self.node_list_rev = topological_order(node_list, in_nodes, out_nodes, rev=True)

        # Now we can store everything -- before calling super constructor,
        # nn.Module doesn't allow assigning anything
        self.in_nodes = in_nodes
        self.condition_nodes = condition_nodes
        self.out_nodes = out_nodes
        self.ff_nodes = ff_nodes

        self.global_out_shapes = global_out_shapes
        self.force_tuple_output = force_tuple_output
        self.module_list = nn.ModuleList([n.module for n in self.node_list_fwd
                                          if n.module is not None])

        if verbose:
            print(self)

    @property
    def node_list(self):
        return self.node_list_fwd

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
        node_list = self.node_list_rev if rev else self.node_list_fwd
        for node in node_list:
            # Skip input/condition/output nodes, they are handled above
            if node in self.in_nodes + self.out_nodes + self.condition_nodes:
                continue

            # Collect inputs to node
            mod_in = []
            mod_c = []
            for prev_node, channel in (node.outputs if rev else node.inputs):
                mod_in.append(outs[prev_node, channel])
            for cond_node, channel in (node.rev_conditions() if rev else node.conditions):
                mod_c.append(outs[cond_node, channel])
            mod_in = tuple(mod_in)
            mod_c = tuple(mod_c)

            try:
                # Execute node
                out, mod_jac = node.forward(x_or_z=mod_in, c=mod_c, rev=rev, jac=jac)
                if jac and mod_jac is not None:
                    jacobian = jacobian + mod_jac
                    jacobian_dict[node] = mod_jac
            except Exception as e:
                raise RuntimeError(f"{node} encountered an error.") from e

            for out_idx, out_value in enumerate(out):
                outs[node, out_idx] = out_value

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


def topological_order(all_nodes: List[AbstractNode], in_nodes: List[InputNode],
                      out_nodes: List[OutputNode], rev: bool) -> List[AbstractNode]:
    """
    Computes the topological order of nodes.

    Parameters:
        all_nodes: All nodes in the computation graph.
        in_nodes: Input nodes (must also be present in `all_nodes`)
        out_nodes: Output nodes (must also be present in `all_nodes`)
        rev: Forward or backward topological order (differs because of conditioning)

    Returns:
        A sorted list of nodes, where the inputs to some node in the list
        are available when all previous nodes in the list have been executed.
    """
    # Topological order differs depending on computation direction
    if not rev:
        edges_out_to_in = {
            node_b: {node_a for node_a, out_idx in node_b.inputs + node_b.conditions} for
            node_b in all_nodes
        }
        start_nodes = in_nodes
        end_nodes = out_nodes
    else:
        edges_out_to_in = {
            node_b: {node_a for node_a, out_idx in node_b.outputs + node_b.rev_conditions()} for
            node_b in all_nodes
        }
        start_nodes = out_nodes
        end_nodes = in_nodes
    # Reverse dict
    edges_in_to_out = defaultdict(set)
    for node_out, node_ins in edges_out_to_in.items():
        for node_in in node_ins:
            edges_in_to_out[node_in].add(node_out)

    # Kahn's algorithm starting from the output nodes
    sorted_nodes = []
    no_pending_edges = deque(end_nodes)

    while len(no_pending_edges) > 0:
        node = no_pending_edges.popleft()
        sorted_nodes.append(node)
        for in_node in list(edges_out_to_in[node]):
            # Mark edge as handled
            edges_out_to_in[node].remove(in_node)
            edges_in_to_out[in_node].remove(node)

            # If this was the last edge to in_node, mark as ready to handle
            if len(edges_in_to_out[in_node]) == 0:
                no_pending_edges.append(in_node)

    for in_node in start_nodes:
        if in_node not in sorted_nodes:
            raise ValueError(f"Error in graph: {in_node} is not connected "
                             f"to any {'out' if not rev else 'in'}put.")

    if sum(map(len, edges_in_to_out.values())) == 0:
        return sorted_nodes[::-1]
    else:
        raise ValueError("Graph is cyclic.")
