'''The framework module contains the logic used in building the graph and
inferring the order that the nodes have to be executed in forward and backward
direction.'''

import sys
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from . import dummy_modules as dummys


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
            self.conditions = [conditions,]

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
                return [inputs,]
            else:
                raise RuntimeError(f"Cannot parse inputs provided to node '{self.name}'.")
        else:
            assert isinstance(inputs, Node), "Received object of invalid type "\
                f"({type(inputs)}) as input for node '{name}'."
            return [(inputs, 0),]

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


class ReversibleGraphNet(nn.Module):
    '''This class represents the invertible net itself. It is a subclass of
    torch.nn.Module and supports the same methods. The forward method has an
    additional option 'rev', whith which the net can be computed in reverse.'''

    def __init__(self, node_list, ind_in=None, ind_out=None, verbose=True):
        '''node_list should be a list of all nodes involved, and ind_in,
        ind_out are the indexes of the special nodes InputNode and OutputNode
        in this list.'''
        super().__init__()

        # Gather lists of input, output and condition nodes
        if ind_in is not None:
            warnings.warn("Use of 'ind_in' and 'ind_out' for ReversibleGraphNet is deprecated, " +
                          "input and output nodes are detected automatically.")
            if isinstance(ind_in, int):
                self.ind_in = list([ind_in])
            else:
                self.ind_in = ind_in
        else:
            self.ind_in = [i for i in range(len(node_list))
                           if isinstance(node_list[i], InputNode)]
            assert len(self.ind_in) > 0, "No input nodes specified."
        if ind_out is not None:
            warnings.warn("Use of 'ind_in' and 'ind_out' for ReversibleGraphNet is deprecated, " +
                          "input and output nodes are detected automatically.")
            if isinstance(ind_out, int):
                self.ind_out = list([ind_out])
            else:
                self.ind_out = ind_out
        else:
            self.ind_out = [i for i in range(len(node_list))
                            if isinstance(node_list[i], OutputNode)]
            assert len(self.ind_out) > 0, "No output nodes specified."
        self.ind_cond = [i for i in range(len(node_list))
                         if isinstance(node_list[i], ConditionNode)]

        self.return_vars = []
        self.input_vars = []
        self.cond_vars = []

        # Assign each node a unique ID
        self.node_list = node_list
        for i, n in enumerate(node_list):
            n.id = i
            n.graph = self

        # Recursively build the nodes nn.Modules and determine order of
        # operations
        ops = []
        for i in self.ind_out:
            node_list[i].build_modules(verbose=verbose)
            node_list[i].run_forward(ops)

        # create list of Pytorch variables that are used
        variables = set()
        for o in ops:
            variables = variables.union(set(o[1] + o[2] + o[3]))
        self.variables_ind = list(variables)

        self.indexed_ops = self.ops_to_indexed(ops)

        self.module_list = nn.ModuleList([n.module for n in node_list])
        self.module_cond = [(len(n.conditions) > 0) for n in node_list]
        self._buffers = {F'tmp_var_{i}' : None for i in range(len(variables))}

        # Find out the order of operations for reverse calculations
        ops_rev = []
        for i in self.ind_in + self.ind_cond:
            node_list[i].run_backward(ops_rev)
        self.indexed_ops_rev = self.ops_to_indexed(ops_rev)

    def ops_to_indexed(self, ops):
        '''Helper function to translate the list of variables (origin ID, channel),
        to variable IDs.'''
        result = []

        for o in ops:
            try:
                vars_in = [self.variables_ind.index(v) for v in o[1]]
            except ValueError:
                vars_in = -1

            vars_out = [self.variables_ind.index(v) for v in o[2]]
            vars_cond = [self.variables_ind.index(v) for v in o[3]]

            # Collect input/output/conditioning nodes in separate lists, but don't
            # add to indexed ops
            if o[0] in self.ind_out:
                self.return_vars.append(self.variables_ind.index(o[1][0]))
                continue
            if o[0] in self.ind_in:
                self.input_vars.append(self.variables_ind.index(o[1][0]))
                continue
            if o[0] in self.ind_cond:
                if self.variables_ind.index(o[1][0]) not in self.cond_vars:
                    self.cond_vars.append(self.variables_ind.index(o[1][0]))
                else:
                    print('Is this branch ever reached?')
                continue

            result.append((o[0], vars_in, vars_out, vars_cond))

        # Sort input/output/conditioning variables so they correspond to initial
        # node list order
        self.return_vars.sort(key=lambda i: self.variables_ind[i][0])
        self.input_vars.sort(key=lambda i: self.variables_ind[i][0])
        self.cond_vars.sort(key=lambda i: self.variables_ind[i][0])

        return result

    def forward(self, x, c=None, rev=False, intermediate_outputs=False):
        '''Forward or backward computation of the whole net.'''

        if rev:
            use_list = self.indexed_ops_rev
            input_vars, output_vars = self.return_vars, self.input_vars
        else:
            use_list = self.indexed_ops
            input_vars, output_vars = self.input_vars, self.return_vars

        # Assign input data to respective variables
        if isinstance(x, (list, tuple)):
            assert len(x) == len(input_vars), (
                f"Got list of {len(x)} input tensors for "
                f"{'inverse' if rev else 'forward'} pass, but expected "
                f"{len(input_vars)}."
            )
            for i in range(len(input_vars)):
                self._buffers[F'tmp_var_{input_vars[i]}'] = x[i]
        else:
            assert len(input_vars) == 1, (f"Got single input tensor for "
                                          f"{'inverse' if rev else 'forward'} "
                                          f"pass, but expected list of "
                                          f"{len(input_vars)}.")
            self._buffers[F'tmp_var_{input_vars[0]}'] = x

        # Assign conditioning data to respective variables
        if c is None:
            assert len(self.cond_vars) == 0
        elif isinstance(c, (list, tuple)):
            assert len(c) == len(self.cond_vars), f'{len(c)}, {len(self.cond_vars)}'
            for i in range(len(self.cond_vars)):
                self._buffers[F'tmp_var_{self.cond_vars[i]}'] = c[i]
        else:
            assert len(self.cond_vars) == 1
            self._buffers[F'tmp_var_{self.cond_vars[0]}'] = c

        # Prepare dictionary for intermediate node outputs
        out_dict = {}

        # Run all modules with the given inputs
        for o in use_list:
            try:
                x = [self._buffers[F'tmp_var_{i}'] for i in o[1]]
                if self.module_cond[o[0]]:
                    c = [self._buffers[F'tmp_var_{i}'] for i in o[3]]
                    results = self.module_list[o[0]](x, c=c, rev=rev)
                else:
                    results = self.module_list[o[0]](x, rev=rev)
            except TypeError:
                print("Are you sure all used Nodes are in the Node list?", file=sys.stderr)
                raise
            out_dict[self.node_list[o[0]].name] = results
            for i, r in zip(o[2], results):
                self._buffers[F'tmp_var_{i}'] = r

        if intermediate_outputs:
            return out_dict
        else:
            out = [self._buffers[F'tmp_var_{output_vars[i]}']
                   for i in range(len(output_vars))]
            if len(out) == 1:
                return out[0]
            else:
                return out

    def log_jacobian(self, x=None, c=None, rev=False, run_forward=True, intermediate_outputs=False):
        '''Compute the log jacobian determinant of the whole net.'''
        if run_forward or c is not None:
            self.condition = c
        jacobian = 0

        if rev:
            use_list = self.indexed_ops_rev
        else:
            use_list = self.indexed_ops

        if run_forward:
            if x is None:
                raise RuntimeError("You need to provide an input if you want "
                                   "to run a forward pass")
            self.forward(x, c, rev=rev)

        # Prepare dictionary for intermediate node outputs
        jacobian_dict = {}

        # Run all modules with the given inputs
        for o in use_list:
            x = [self._buffers[F'tmp_var_{i}'] for i in o[1]]
            if self.module_cond[o[0]]:
                c = [self._buffers[F'tmp_var_{i}'] for i in o[3]]
                module_jacobian = self.module_list[o[0]].jacobian(x, c=c, rev=rev)
            else:
                module_jacobian = self.module_list[o[0]].jacobian(x, rev=rev)
            jacobian += module_jacobian
            jacobian_dict[self.node_list[o[0]].name] = module_jacobian

        if intermediate_outputs:
            return jacobian_dict
        else:
            return jacobian

    def jacobian(self, *args, **kwargs):
        '''Compute the log jacobian determinant of the whole net.'''
        warnings.warn("This function computes the log-jacobian determinant, not the "
                      "jacobian as the name suggest. Will be removed in the future.")
        return self.log_jacobian(*args, **kwargs)

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
            offset[:,i] = h
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
            J_num[:,:,i] = (y_upper - y_lower).view(batch_size, -1) / (2*h)
        logdet_num = x[0].new_zeros(batch_size)
        for i in range(batch_size):
            logdet_num[i] = torch.det(J_num[i,:,:]).abs().log()

        return logdet_num

    def load_state_dict(self, state_dict, *args, **kwargs):

        state_dict_no_buffers = {}
        for k,p in state_dict.items():
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



# Testing example
if __name__ == '__main__':
    inp = InputNode(4, 64, 64, name='input')
    t1 = Node([(inp, 0)], dummys.dummy_mux, {}, name='t1')
    s1 = Node([(t1, 0)], dummys.dummy_2split, {}, name='s1')

    t2 = Node([(s1, 0)], dummys.dummy_module, {}, name='t2')
    s2 = Node([(s1, 1)], dummys.dummy_2split, {}, name='s2')
    t3 = Node([(s2, 0)], dummys.dummy_module, {}, name='t3')

    m1 = Node([(t3, 0), (s2, 1)], dummys.dummy_2merge, {}, name='m1')
    m2 = Node([(t2, 0), (m1, 0)], dummys.dummy_2merge, {}, name='m2')
    outp = OutputNode([(m2, 0)], name='output')

    all_nodes = [inp, outp, t1, s1, t2, s2, t3, m1, m2]

    net = ReversibleGraphNet(all_nodes, 0, 1)
