import unittest

import torch
import torch.nn as nn
import numpy as np

import FrEIA.modules as Fm
import FrEIA.framework as Ff
from FrEIA.framework import collect_nodes


def F_conv(cin, cout):
    '''Simple convolutional subnetwork'''
    net = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, cout, 3, padding=1))
    net.apply(subnet_initialization)
    return net


def F_fully_connected(cin, cout):
    '''Simple fully connected subnetwork'''
    net = nn.Sequential(nn.Linear(cin, 128),
                        nn.ReLU(),
                        nn.Linear(128, cout))
    net.apply(subnet_initialization)
    return net


# the reason the subnet init is needed, is that with uninitalized
# weights, the numerical jacobian check gives inf, nan, etc,
def subnet_initialization(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        m.weight.data *= 0.3
        m.bias.data *= 0.1


class SimpleComputeGraph(unittest.TestCase):

    def test_build(self):

        in_node = Ff.InputNode(3, 10, 10)
        out_node = Ff.OutputNode(in_node)
        graph = Ff.GraphINN([in_node, out_node])

        # the input node should not have any graph edges going in
        self.assertEqual(in_node.input_dims, [])
        # the output node should not have any graph edges going out
        self.assertEqual(out_node.output_dims, [])

        # dimensions should match
        self.assertEqual(in_node.output_dims, out_node.input_dims)
        self.assertEqual(graph.dims_in, in_node.output_dims)


class ComplexComputeGraph(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)

        self.inp_size = (3, 10, 10)
        self.cond_size = (1, 10, 10)

        inp = Ff.InputNode(*self.inp_size, name='input')
        cond = Ff.ConditionNode(*self.cond_size, name='cond')
        split = Ff.Node(inp, Fm.Split, {'section_sizes': [1,2], 'dim': 0}, name='split1')

        flatten1 = Ff.Node(split.out0, Fm.Flatten, {}, name='flatten1')
        perm = Ff.Node(flatten1, Fm.PermuteRandom, {'seed': 123}, name='perm')
        unflatten1 = Ff.Node(perm, Fm.Reshape, {'output_dims': (1, 10, 10)}, name='unflatten1')

        conv = Ff.Node(split.out1,
                       Fm.RNVPCouplingBlock,
                       {'subnet_constructor': F_conv, 'clamp': 1.0},
                       conditions=cond,
                       name='conv')

        flatten2 = Ff.Node(conv, Fm.Flatten, {}, name='flatten2')

        linear = Ff.Node(flatten2,
                         Fm.RNVPCouplingBlock,
                         {'subnet_constructor': F_fully_connected, 'clamp': 1.0},
                         name='linear')

        unflatten2 = Ff.Node(linear, Fm.Reshape, {'output_dims': (2, 10, 10)}, name='unflatten2')
        concat = Ff.Node([unflatten1.out0, unflatten2.out0], Fm.Concat, {'dim': 0}, name='concat')
        haar = Ff.Node(concat, Fm.HaarDownsampling, {}, name='haar')
        out = Ff.OutputNode(haar, name='output')

        self.test_net = Ff.GraphINN([inp, cond, split, flatten1, perm, unflatten1, conv,
                                     flatten2, linear, unflatten2, concat, haar, out])

        # this is only used for the cuda variant of the tests.
        # if true, all tests are skipped.
        self.skip_all = False

        self.batch_size = 32
        self.inv_tol = 1e-4
        torch.manual_seed(self.batch_size)

        self.x = torch.randn(self.batch_size, *self.inp_size)
        self.cond = torch.randn(self.batch_size, *self.cond_size)

    def test_output_shape(self):

        if self.skip_all:
            raise unittest.SkipTest("No CUDA-device found, skipping CUDA test.")

        y = self.test_net(self.x, c=[self.cond])[0]
        self.assertTrue(isinstance(y, type(self.x) ), f"{type(y)}")

        exp = torch.Size([self.batch_size, self.inp_size[0]*4, self.inp_size[1]//2, self.inp_size[2]//2])
        self.assertEqual(y.shape, exp , f"{y.shape}")

    def test_inverse(self):

        if self.skip_all:
            raise unittest.SkipTest("No CUDA-device found, skipping CUDA test.")

        y, j = self.test_net(self.x, c=[self.cond])
        x_re, j_re = self.test_net(y, c=[self.cond], rev=True)

        obs = torch.max(torch.abs(self.x - x_re))
        obs_j = torch.max(torch.abs(j + j_re))
        self.assertTrue(obs < self.inv_tol, f"Inversion {obs} !< {self.inv_tol}")
        self.assertTrue(obs_j  < self.inv_tol, f"Jacobian inversion {obs} !< {self.inv_tol}")

    def test_jacobian(self):

        if self.skip_all:
            raise unittest.SkipTest("No CUDA-device found, skipping CUDA test.")

        logdet = self.test_net(self.x, c=[self.cond])[1]
        # Approximate log det of Jacobian numerically
        logdet_num = self.test_net.log_jacobian_numerical(self.x, c=[self.cond], h=1e-3)
        # Check that they are the same (within tolerance)
        obs = torch.allclose(logdet, logdet_num, atol=np.inf, rtol=0.03)
        self.assertTrue(obs, f"Numerical Jacobian check {logdet, logdet_num}")


class ComplexComputeGraphCuda(ComplexComputeGraph):

    def __init__(self, *args):
        super().__init__(*args)

        if torch.cuda.is_available():
            self.x = self.x.cuda()
            self.cond = self.cond.cuda()
            self.test_net.cuda()
        else:
            self.skip_all = True


class GraphTopology(unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.inp1_size = (4, 10, 10)
        self.inp2_size = (4, 10, 10)
        self.batch_size = 32
        torch.manual_seed(self.batch_size)

        self.inp1 = torch.randn(self.batch_size, *self.inp1_size)
        self.inp2 = torch.randn(self.batch_size, *self.inp2_size)

    def test_cyclic_graph(self):
        input_node = Ff.InputNode(*self.inp1_size, name="input")
        core_cond = Ff.FeedForwardNode(input_node, (4, 10, 10), nn.Identity, name="core_cond")
        coupled = Ff.Node(input_node, Fm.RNVPCouplingBlock,
                          {'subnet_constructor': F_conv, 'clamp': 1.0},
                          core_cond,
                          name='coupled')

        Ff.OutputNode(coupled, name="out")
        with self.assertRaises(ValueError):
            Ff.GraphINN(collect_nodes(input_node))

    def test_double_ff_graph(self):
        input_node = Ff.InputNode(*self.inp1_size, name="input1")
        split = Ff.Node(input_node, Fm.Split, dict(section_sizes=2), name="split")
        uncoupled = Ff.Node(split, Fm.RNVPCouplingBlock,
                            {'subnet_constructor': F_conv, 'clamp': 1.0},
                            name="uncoupled")
        cond1 = Ff.FeedForwardNode(split, split.output_dims[0], nn.Identity, name="cond1")
        cond2 = Ff.FeedForwardNode(split, split.output_dims[0], nn.Identity, name="cond2")
        coupled1 = Ff.Node(split.out1, Fm.RNVPCouplingBlock,
                          {'subnet_constructor': F_conv, 'clamp': 1.0},
                          cond1,
                          name='coupled1')
        coupled2 = Ff.Node(coupled1, Fm.RNVPCouplingBlock,
                          {'subnet_constructor': F_conv, 'clamp': 1.0},
                          cond2,
                          name='coupled2')
        merge = Ff.Node([uncoupled.out0, coupled2], Fm.Concat, name="merge")
        out = Ff.OutputNode(merge, name="out")

        inn = Ff.GraphINN(collect_nodes(input_node))

        y = inn(self.inp1)[0]
        self.assertTrue(isinstance(y, type(self.inp1)), f"{type(y)}")

        # the input node should not have any graph edges going in
        self.assertEqual(input_node.input_dims, [])
        # the output node should not have any graph edges going out
        self.assertEqual(out.output_dims, [])

        # dimension of output should match spec
        self.assertEqual(y.shape[1:], out.input_dims[0])


if __name__ == '__main__':
    unittest.main()
