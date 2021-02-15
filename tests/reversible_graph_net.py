import unittest

import torch
import torch.nn as nn
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import sys
sys.path.append('../')
from FrEIA.modules import *
from FrEIA.framework import *


def F_conv(cin, cout):
    '''Simple convolutional subnetwork'''
    return nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(32, cout, 3, padding=1))

def F_fully_connected(cin, cout):
    '''Simple fully connected subnetwork'''
    return nn.Sequential(nn.Linear(cin, 128),
                         nn.ReLU(),
                         nn.Linear(128, cout))


class ComplexComputeGraph(unittest.TestCase):


    def __init__(self, *args):
        super().__init__(*args)

        self.inp_size = (3, 10, 10)
        self.cond_size = (1, 10, 10)

        inp = InputNode(*self.inp_size, name='input')
        cond = ConditionNode(*self.cond_size, name='cond')

        split = Node(inp,
                     Split,
                     {'section_sizes': [1,2], 'dim': 0},
                     name='split1')

        flatten1 = Node(split.out0, Flatten, {}, name='flatten1')
        perm = Node(flatten1, PermuteRandom, {'seed': 123}, name='perm')
        unflatten1 = Node(perm, Reshape, {'target_dim': (1, 10, 10)}, name='unflatten1')

        conv = Node(split.out1,
                    RNVPCouplingBlock,
                    {'subnet_constructor': F_conv, 'clamp': 2.0},
                    conditions=cond,
                    name='conv')
        flatten2 = Node(conv, Flatten, {}, name='flatten2')
        linear = Node(flatten2,
                      RNVPCouplingBlock,
                      {'subnet_constructor': F_fully_connected, 'clamp': 2.0},
                      name='linear')
        unflatten2 = Node(linear, Reshape, {'target_dim': (2, 10, 10)}, name='unflatten2')

        concat = Node([unflatten1.out0, unflatten2.out0],
                      Concat,
                      {'dim': 0},
                      name='concat')
        haar = Node(concat, HaarDownsampling, {}, name='haar')

        out = OutputNode(haar, name='output')
        self.test_net = ReversibleGraphNet([inp, cond, split, flatten1, perm, unflatten1, conv,
            flatten2, linear, unflatten2, concat, haar, out])


        self.batch_size = 32
        self.tol = 1e-4
        torch.manual_seed(self.batch_size)

        self.x = torch.randn(self.batch_size, *self.inp_size).to(DEVICE)
        self.cond = torch.randn(self.batch_size, *self.cond_size).to(DEVICE)

    def test_constructs(self):

        self.test_net.to(DEVICE)
        y = self.test_net(self.x, c=[self.cond]).to(DEVICE)
        self.assertTrue(isinstance(y, type(self.x) ), f"{type(y)}")

        exp = torch.Size([self.batch_size, self.inp_size[0]*4, self.inp_size[1]//2, self.inp_size[2]//2])
        self.assertEqual(y.shape, exp , f"{y.shape}")

    def test_inverse(self):

        self.test_net.to(DEVICE)
        y = self.test_net(self.x, c=[self.cond]).to(DEVICE)
        x_re = self.test_net(y, c=[self.cond], rev=True).to(DEVICE)

        obs = torch.max(torch.abs(self.x - x_re))
        self.assertTrue(obs < self.tol, f"{obs} !< {self.tol}")

    def test_jacobian(self):

        # Compute log det of Jacobian
        self.test_net.to(DEVICE)
        y = self.test_net(self.x, c=[self.cond])
        y.to(DEVICE)
        logdet = self.test_net.log_jacobian( self.x, c=[self.cond] ).to(DEVICE)
        # Approximate log det of Jacobian numerically
        logdet_num = self.test_net.log_jacobian_numerical(self.x, c=[self.cond]).to(DEVICE)
        # Check that they are the same (within tolerance)
        obs = torch.allclose(logdet, logdet_num, atol=0.01, rtol=0.01)
        self.assertTrue(obs, f"{logdet, logdet_num}")

    @unittest.skipIf(not torch.cuda.is_available(),
                     "CUDA capable device not available")
    def test_cuda(self):
        self.test_net.to('cuda')
        x = torch.randn(self.batch_size, *self.inp_size).cuda()
        cond = torch.randn(self.batch_size, *self.cond_size).cuda()

        y = self.test_net(x, c=[cond])
        x_re = self.test_net(y, c=[cond], rev=True)

        obs = torch.max(torch.abs(x - x_re))
        self.assertTrue(obs < self.tol, f"{obs} !< {self.tol}")

        self.test_net.to('cpu')

    def test_on_any(self):
        self.test_net.to(DEVICE)
        x = torch.randn(self.batch_size, *self.inp_size).to(DEVICE)
        cond = torch.randn(self.batch_size, *self.cond_size).to(DEVICE)

        y = self.test_net(x, c=[cond])
        x_re = self.test_net(y, c=[cond], rev=True)

        obs = torch.max(torch.abs(x - x_re))
        self.assertTrue(obs < self.tol, f"{obs} !< {self.tol}")
        self.test_net.to('cpu')



if __name__ == '__main__':
    unittest.main()
