import unittest

import torch
import torch.nn as nn
import torch.optim

import sys
sys.path.append('../')
from FrEIA.modules import *
from FrEIA.framework import *


inp_size = (3, 10, 10)
c1_size = (1, 10, 10)
c2_size = (50,)
c3_size = (20,)

inp = InputNode(*inp_size, name='input')
c1 = ConditionNode(*c1_size, name='c1')
conv = Node(inp,
            rev_multiplicative_layer,
            {'F_class': F_conv, 'clamp': 6.0},
            conditions=c1,
            name='conv')
flatten = Node(conv,
               flattening_layer,
               {},
               name='flatten')
c2 = ConditionNode(*c2_size, name='c2')
c3 = ConditionNode(*c3_size, name='c3')
linear = Node(flatten,
              rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 6.0},
              conditions=[c2,c3],
              name='linear')
outp = OutputNode(linear, name='output')
test_net = ReversibleGraphNet([inp, c1, conv, flatten, c2, c3, linear, outp])

# for name, p in test_net.named_parameters():
#     print(name, p.shape, p.requires_grad)


class ConditioningTest(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)

        self.batch_size = 256
        self.tol = 1e-4
        torch.manual_seed(0)

    def test_inverse(self):
        x = torch.randn(self.batch_size, *inp_size)
        c1 = torch.randn(self.batch_size, *c1_size)
        c2 = torch.randn(self.batch_size, *c2_size)
        c3 = torch.randn(self.batch_size, *c3_size)

        y = test_net(x, c=[c1,c2,c3])
        x_re = test_net(y, c=[c1,c2,c3], rev=True)

        # if torch.max(torch.abs(x - x_re)) > self.tol:
        #     print(torch.max(torch.abs(x - x_re)).item(), end='   ')
        #     print(torch.mean(torch.abs(x - x_re)).item())
        self.assertTrue(torch.max(torch.abs(x - x_re)) < self.tol)

        # Assert that wrong condition inputs throw exceptions
        with self.assertRaises(Exception) as context:
            y = test_net(x, c=[c2,c1,c3])

        c2a = torch.randn(self.batch_size, c2_size[0] + 1, *c2_size[1:])
        # c3a = torch.randn(self.batch_size, c3_size[0] - 1, *c3_size[1:])
        with self.assertRaises(Exception) as context:
            y = test_net(x, c=[c1,c2a,c3])

        c1a = torch.randn(self.batch_size, *c1_size[:2], c1_size[2] + 1)
        with self.assertRaises(Exception) as context:
            y = test_net(x, c=[c1a,c2,c3])


    def test_jacobian(self):
        x = torch.randn(self.batch_size, *inp_size)
        c1 = torch.randn(self.batch_size, *c1_size)
        c2 = torch.randn(self.batch_size, *c2_size)
        c3 = torch.randn(self.batch_size, *c3_size)

        # Compute log det of Jacobian
        y = test_net(x, c=[c1,c2,c3])
        logdet = test_net.log_jacobian(x, c=[c1,c2,c3])
        # Approximate log det of Jacobian numerically
        logdet_num = test_net.log_jacobian_numerical(x, c=[c1,c2,c3])
        # Check that they are the same (within tolerance)
        self.assertTrue(torch.allclose(logdet, logdet_num, atol=0.01, rtol=0.01))


    def test_cuda(self):
        test_net.to('cuda')
        x = torch.randn(self.batch_size, *inp_size).cuda()
        c1 = torch.randn(self.batch_size, *c1_size).cuda()
        c2 = torch.randn(self.batch_size, *c2_size).cuda()
        c3 = torch.randn(self.batch_size, *c3_size).cuda()

        y = test_net(x, c=[c1,c2,c3])
        x_re = test_net(y, c=[c1,c2,c3], rev=True)

        self.assertTrue(torch.max(torch.abs(x - x_re)) < self.tol)
        test_net.to('cpu')


if __name__ == '__main__':
    unittest.main()
