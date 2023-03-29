import unittest

import torch
import torch.nn as nn

from FrEIA.framework import GraphINN, InputNode, Node, OutputNode, ConditionNode, collect_nodes
from FrEIA.modules import AllInOneBlock, Split, Reshape, Flatten, RNVPCouplingBlock, PermuteRandom, HaarDownsampling, Concat
from FrEIA.utils import plot_graph

import os

# the reason the subnet init is needed, is that with uninitalized
# weights, the numerical jacobian check gives inf, nan, etc,
def subnet_initialization(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        m.weight.data *= 0.3
        m.bias.data *= 0.1

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

class GraphINNTest(unittest.TestCase):
    def test_existing_module(self):
        nodes = []
        dim = 3

        nodes.append(InputNode(dim))
        nodes.append(Node(nodes[-1], AllInOneBlock(nodes[-1].output_dims, subnet_constructor=torch.nn.Linear)))
        nodes.append(OutputNode(nodes[-1]))
        graph_inn = GraphINN(nodes)

        batch_size = 16
        out, jac = graph_inn(torch.randn(batch_size, dim))
        assert out.shape == (batch_size, dim)

class PlotGraphINNTest(unittest.TestCase):
    plotdir = os.path.join(os.getcwd(),"graphINN_test_plots")
    plot_name = "graph"
    file_path = os.path.join(plotdir, plot_name)
    has_graphviz_backend = True

    def cleanup_files(self):
        if os.path.exists(self.file_path):
                os.remove(self.file_path)
        if os.path.exists(self.file_path + ".pdf"):
            os.remove(self.file_path + ".pdf")

    @classmethod
    def setUpClass(self):
        os.mkdir(self.plotdir)

        in_node = InputNode(3, 10, 10)
        out_node = OutputNode(in_node)
        try:
            plot_graph([in_node, out_node], path=self.plotdir, filename=self.plot_name)
        except Exception:
            self.cleanup_files(self)
            self.has_graphviz_backend = False
        self.cleanup_files(self)
        
    @classmethod
    def tearDownClass(self) -> None:
        os.rmdir(self.plotdir)

    def setUp(self):
        if not self.has_graphviz_backend:
            self.skipTest('Skipped testing graph plots since graphviz backend is not installed.')

    def tearDown(self) -> None:
        self.cleanup_files()

    def test_input_output_graph(self):
        in_node = InputNode(3, 10, 10)
        out_node = OutputNode(in_node)
        graph = GraphINN([in_node, out_node])
        graph.plot(path=self.plotdir, filename=self.plot_name)

        self.assertTrue(os.path.exists(self.file_path))
        self.assertTrue(os.path.exists(self.file_path + ".pdf"))

    def test_raises_non_existing_path(self):
        in_node = InputNode(3, 10, 10)
        out_node = OutputNode(in_node)
        graph = GraphINN([in_node, out_node])

        self.assertRaises(Exception, graph.plot, "not_existing_path", self.plot_name)

    def test_one_layer_graph(self):
        nodes = []
        dim = 3
        nodes.append(InputNode(dim))
        nodes.append(Node(nodes[-1], AllInOneBlock(nodes[-1].output_dims, subnet_constructor=torch.nn.Linear)))
        nodes.append(OutputNode(nodes[-1]))
        graph = GraphINN(nodes)
        graph.plot(path=self.plotdir, filename=self.plot_name)

        self.assertTrue(os.path.exists(self.file_path))
        self.assertTrue(os.path.exists(self.file_path + ".pdf"))

    def test_complex_graph(self):
        inp_size = (3, 10, 10)
        cond_size = (1, 10, 10)

        inp = InputNode(*inp_size, name='input')
        cond = ConditionNode(*cond_size, name='cond')
        split = Node(inp, Split, {'section_sizes': [1,2], 'dim': 0}, name='split1')

        flatten1 = Node(split.out0, Flatten, {}, name='flatten1')
        perm = Node(flatten1, PermuteRandom, {'seed': 123}, name='perm')
        unflatten1 = Node(perm, Reshape, {'output_dims': (1, 10, 10)}, name='unflatten1')

        conv = Node(split.out1,
                    RNVPCouplingBlock,
                    {'subnet_constructor': F_conv, 'clamp': 1.0},
                    conditions=cond,
                    name='conv')

        flatten2 = Node(conv, Flatten, {}, name='flatten2')

        linear = Node(flatten2,
                      RNVPCouplingBlock,
                      {'subnet_constructor': F_fully_connected, 'clamp': 1.0},
                      name='linear')

        unflatten2 = Node(linear, Reshape, {'output_dims': (2, 10, 10)}, name='unflatten2')
        concat = Node([unflatten1.out0, unflatten2.out0], Concat, {'dim': 0}, name='concat')
        haar = Node(concat, HaarDownsampling, {}, name='haar')
        out = OutputNode(haar, name='output')

        auto_node_list = collect_nodes(inp)
        manual_node_list = [inp, cond, split, flatten1, perm, unflatten1, conv, flatten2, linear, unflatten2, concat, haar, out]
        self.assertEquals(set(auto_node_list), set(manual_node_list))
        plot_graph(auto_node_list, path=self.plotdir, filename=self.plot_name)

        self.assertTrue(os.path.exists(self.file_path))
        self.assertTrue(os.path.exists(self.file_path + ".pdf"))


if __name__ == '__main__':
    unittest.main()