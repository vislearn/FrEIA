'''The framework module contains the logic used in building the graph and
inferring the order that the nodes have to be executed in forward and backward
direction.'''

import torch.nn as nn
from torch.autograd import Variable

import FrEIA.dummy_modules as dummys


class Node:
    '''The Node class represents one transformation in the graph, with an
    arbitrary number of in- and outputs.'''
    def __init__(self, inputs, module_type, module_args, name=None):
        self.inputs = inputs
        self.outputs = []
        self.module_type = module_type
        self.module_args = module_args

        self.input_dims, self.module = None, None
        self.computed = None
        self.computed_rev = None
        self.id = None

        if name:
            self.name = name
        else:
            self.name = hex(id(self))[-6:]
        for i in range(255):
            exec('self.out{0} = (self, {0})'.format(i))

    def build_modules(self, verbose=True):
        ''' Returns a list with the dimension of each output of this node,
        recursively calling build_modules of the nodes connected to the input.
        Use this information to initialize the pytorch nn.Module of this node.
        '''

        if not self.input_dims:  # Only do it if this hasn't been computed yet
            self.input_dims = [n.build_modules(verbose=verbose)[c]
                               for n, c in self.inputs]
            try:
                self.module = self.module_type(self.input_dims,
                                               **self.module_args)
            except Exception as e:
                print('Error in node %s' % (self.name))
                raise e

            if verbose:
                print("Node %s has following input dimensions:" % (self.name))
                for d, (n, c) in zip(self.input_dims, self.inputs):
                    print("\t Output #%i of node %s:" % (c, n.name), d)
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
                # Register youself as an output in the input node
                n.outputs.append((self, i))

            # All outputs could now be computed
            self.computed = [(self.id, i) for i in range(self.n_outputs)]
            op_list.append((self.id, self.input_vars, self.computed))

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
            op_list.append((self.id, output_vars, self.computed_rev))

        return self.computed_rev


class InputNode(Node):
    '''Special type of node that represents the input data of the whole net (or
    ouput when running reverse)'''

    def __init__(self, *dims, name='node'):
        self.name = name
        self.data = dummys.dummy_data(*dims)
        self.outputs = []
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
            super(OutputNode.dummy, self).__init__()

        def __call__(*args):
            return args

        def output_dims(*args):
            return args

    def __init__(self, inputs, name='node'):
        self.module_type, self.module_args = self.dummy, {}
        self.output_dims = []
        self.inputs = inputs
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
        super(ReversibleGraphNet, self).__init__()

        # Gather lists of input and output nodes
        if ind_in is not None:
            if isinstance(ind_in, int):
                self.ind_in = list([ind_in])
            else:
                self.ind_in = ind_in
        else:
            self.ind_in = [i for i in range(len(node_list))
                           if isinstance(node_list[i], InputNode)]
            assert len(self.ind_in) > 0, "No input nodes specified."
        if ind_out is not None:
            if isinstance(ind_out, int):
                self.ind_out = list([ind_out])
            else:
                self.ind_out = ind_out
        else:
            self.ind_out = [i for i in range(len(node_list))
                            if isinstance(node_list[i], OutputNode)]
            assert len(self.ind_out) > 0, "No output nodes specified."

        self.return_vars = []
        self.input_vars = []

        # Assign each node a unique ID
        self.node_list = node_list
        for i, n in enumerate(node_list):
            n.id = i

        # Recursively build the nodes nn.Modules and determine order of
        # operations
        ops = []
        for i in self.ind_out:
            node_list[i].build_modules(verbose=verbose)
            node_list[i].run_forward(ops)

        # create list of Pytorch variables that are used
        variables = set()
        for o in ops:
            variables = variables.union(set(o[1] + o[2]))
        self.variables_ind = list(variables)

        self.indexed_ops = self.ops_to_indexed(ops)

        self.module_list = nn.ModuleList([n.module for n in node_list])
        self.variable_list = [Variable(requires_grad=True) for v in variables]

        # Find out the order of operations for reverse calculations
        ops_rev = []
        for i in self.ind_in:
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

            # Collect input/output nodes in separate lists, but don't add to
            # indexed ops
            if o[0] in self.ind_out:
                self.return_vars.append(self.variables_ind.index(o[1][0]))
                continue
            if o[0] in self.ind_in:
                self.input_vars.append(self.variables_ind.index(o[1][0]))
                continue

            result.append((o[0], vars_in, vars_out))

        # Sort input/output variables so they correspond to initial node list
        # order
        self.return_vars.sort(key=lambda i: self.variables_ind[i][0])
        self.input_vars.sort(key=lambda i: self.variables_ind[i][0])

        return result

    def forward(self, x, rev=False):
        '''Forward or backward computation of the whole net.'''
        if rev:
            use_list = self.indexed_ops_rev
            input_vars, output_vars = self.return_vars, self.input_vars
        else:
            use_list = self.indexed_ops
            input_vars, output_vars = self.input_vars, self.return_vars

        if isinstance(x, (list, tuple)):
            assert len(x) == len(input_vars), (
                f"Got list of {len(x)} input tensors for "
                f"{'inverse' if rev else 'forward'} pass, but expected "
                f"{len(input_vars)}."
            )
            for i in range(len(input_vars)):
                self.variable_list[input_vars[i]] = x[i]
        else:
            assert len(input_vars) == 1, (f"Got single input tensor for "
                                          f"{'inverse' if rev else 'forward'} "
                                          f"pass, but expected list of "
                                          f"{len(input_vars)}.")
            self.variable_list[input_vars[0]] = x

        for o in use_list:
            try:
                results = self.module_list[o[0]]([self.variable_list[i]
                                                  for i in o[1]], rev=rev)
            except TypeError:
                raise RuntimeError("Are you sure all used Nodes are in the "
                                   "Node list?")
            for i, r in zip(o[2], results):
                self.variable_list[i] = r
            # self.variable_list[o[2][0]] = self.variable_list[o[1][0]]

        out = [self.variable_list[output_vars[i]]
               for i in range(len(output_vars))]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def jacobian(self, x=None, rev=False, run_forward=True):
        '''Compute the jacobian determinant of the whole net.'''
        jacobian = 0

        if rev:
            use_list = self.indexed_ops_rev
        else:
            use_list = self.indexed_ops

        if run_forward:
            if x is None:
                raise RuntimeError("You need to provide an input if you want "
                                   "to run a forward pass")
            self.forward(x, rev=rev)

        for o in use_list:
            try:
                jacobian += self.module_list[o[0]].jacobian(
                    [self.variable_list[i] for i in o[1]], rev=rev
                )
            except TypeError:
                raise RuntimeError("Are you sure all used Nodes are in the "
                                   "Node list?")

        return jacobian


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
