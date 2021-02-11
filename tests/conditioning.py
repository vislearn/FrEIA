import unittest

import torch
import torch.nn as nn
import torch.optim
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import sys
sys.path.append('../')
from FrEIA.modules import *
from FrEIA.framework import *

inp_size = (3, 10, 10)
c1_size = (1, 10, 10)
c2_size = (50,)
c3_size = (20,)

def F_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(32, cout, 3, padding=1))

def F_fully_connected(cin, cout):
    return nn.Sequential(nn.Linear(cin, 128),
                         nn.ReLU(),
                         nn.Linear(128, cout))


inp = InputNode(*inp_size, name='input')
c1 = ConditionNode(*c1_size, name='c1')
conv = Node(inp,
            RNVPCouplingBlock,
            {'subnet_constructor': F_conv, 'clamp': 6.0},
            conditions=c1,
            name='conv::c1')
flatten = Node(conv,
               Flatten,#flattening_layer,
               {},
               name='flatten')
c2 = ConditionNode(*c2_size, name='c2')
c3 = ConditionNode(*c3_size, name='c3')
linear = Node(flatten,
              RNVPCouplingBlock,
            {'subnet_constructor':F_fully_connected, 'clamp': 6.0},
              conditions=[c2,c3],
              name='linear::c2|c3')
outp = OutputNode(linear, name='output')
conv_outp = OutputNode(conv, name='output')
test_net = ReversibleGraphNet([inp, c1, conv, flatten, c2, c3, linear, outp])


class ConditioningTest(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)

        self.batch_size = 32
        self.tol = 1e-4
        torch.manual_seed(self.batch_size)

        self.x = torch.randn(self.batch_size, *inp_size).to(DEVICE)
        self.c1 = torch.randn(self.batch_size, *c1_size).to(DEVICE)
        self.c2 = torch.randn(self.batch_size, *c2_size).to(DEVICE)
        self.c3 = torch.randn(self.batch_size, *c3_size).to(DEVICE)

    def test_constructs(self):

        y = test_net(self.x, c=[self.c1,self.c2,self.c3]).to(DEVICE)
        self.assertTrue(isinstance(y, type(self.x) ), f"{type(y)}")

        exp = torch.Size([self.batch_size, inp_size[0]*inp_size[1]*inp_size[2]])
        self.assertEqual(y.shape, exp , f"{y.shape}")

    def test_inverse(self):

        y = test_net(self.x, c=[self.c1,self.c2,self.c3]).to(DEVICE)
        x_re = test_net(y, c=[self.c1,self.c2,self.c3], rev=True).to(DEVICE)

        # if torch.max(torch.abs(x - x_re)) > self.tol:
        #     print(torch.max(torch.abs(x - x_re)).item(), end='   ')
        #     print(torch.mean(torch.abs(x - x_re)).item())
        obs = torch.max(torch.abs(self.x - x_re))
        self.assertTrue(obs < self.tol, f"{obs} !< {self.tol}")

        # Assert that wrong condition inputs throw exceptions
        with self.assertRaises(Exception) as context:
            y = test_net(self.x, c=[self.c2,self.c1,self.c3]).to(DEVICE)

        c2a = torch.randn(self.batch_size, c2_size[0] + 4, *c2_size[1:]).to(DEVICE)
        # c3a = torch.randn(self.batch_size, c3_size[0] - 1, *c3_size[1:])
        with self.assertRaises(Exception) as context:
            y = test_net(self.x, c=[self.c1,c2a,self.c3])

        c1a = torch.randn(self.batch_size, *c1_size[:2], c1_size[2] + 1).to(DEVICE)
        with self.assertRaises(Exception) as context:
            y = test_net(self.x, c=[c1a,self.c2,self.c3])


    def test_jacobian(self):
        # Compute log det of Jacobian
        test_net.to(DEVICE)
        y = test_net(self.x, c=[self.c1,self.c2,self.c3])
        y.to(DEVICE)
        logdet = test_net.log_jacobian( self.x, c=[self.c1,self.c2,self.c3] ).to(DEVICE)
        # Approximate log det of Jacobian numerically
        logdet_num = test_net.log_jacobian_numerical( self.x, c=[self.c1,self.c2,self.c3] ).to(DEVICE)
        # Check that they are the same (within tolerance)
        obs = torch.allclose(logdet, logdet_num, atol=0.01, rtol=0.01)
        self.assertTrue(obs, f"{logdet, logdet_num}")

    @unittest.skipIf(not torch.cuda.is_available(),
                     "CUDA capable device not available")
    def test_cuda(self):
        test_net.to('cuda')
        x = torch.randn(self.batch_size, *inp_size).cuda()
        c1 = torch.randn(self.batch_size, *c1_size).cuda()
        c2 = torch.randn(self.batch_size, *c2_size).cuda()
        c3 = torch.randn(self.batch_size, *c3_size).cuda()

        y = test_net(x, c=[c1,c2,c3])
        x_re = test_net(y, c=[c1,c2,c3], rev=True)

        obs = torch.max(torch.abs(x - x_re))
        self.assertTrue(obs < self.tol, f"{obs} !< {self.tol}")

        test_net.to('cpu')

    def test_on_any(self):
        test_net.to(DEVICE)
        x = torch.randn(self.batch_size, *inp_size).to(DEVICE)
        c1 = torch.randn(self.batch_size, *c1_size).to(DEVICE)
        c2 = torch.randn(self.batch_size, *c2_size).to(DEVICE)
        c3 = torch.randn(self.batch_size, *c3_size).to(DEVICE)

        y = test_net(x, c=[c1,c2,c3])
        x_re = test_net(y, c=[c1,c2,c3], rev=True)

        obs = torch.max(torch.abs(x - x_re))
        self.assertTrue(obs < self.tol, f"{obs} !< {self.tol}")
        test_net.to('cpu')


if __name__ == '__main__':
    unittest.main()
