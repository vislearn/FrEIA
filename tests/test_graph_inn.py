import unittest

import torch

from FrEIA.framework import GraphINN, InputNode, Node, OutputNode
from FrEIA.modules import AllInOneBlock


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
