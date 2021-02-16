import unittest

import torch
import torch.nn as nn
import torch.optim
import numpy as np

import FrEIA.modules as Fm
import FrEIA.framework as Ff


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
        m.bias.data *= 0.1


class ConditioningTest(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)

        self.batch_size = 32
        self.inv_tol = 1e-4
        torch.manual_seed(self.batch_size)

        self.inp_size = (3, 10, 10)
        self.c1_size = (1, 10, 10)
        self.c2_size = (50,)
        self.c3_size = (20,)

        self.x = torch.randn(self.batch_size, *self.inp_size)
        self.c1 = torch.randn(self.batch_size, *self.c1_size)
        self.c2 = torch.randn(self.batch_size, *self.c2_size)
        self.c3 = torch.randn(self.batch_size, *self.c3_size)

        # this is only used for the cuda variant of the tests.
        # if true, all tests are skipped.
        self.skip_all = False

        inp = Ff.InputNode(*self.inp_size, name='input')
        c1 = Ff.ConditionNode(*self.c1_size, name='c1')
        c2 = Ff.ConditionNode(*self.c2_size, name='c2')
        c3 = Ff.ConditionNode(*self.c3_size, name='c3')

        conv = Ff.Node(inp,
                       Fm.RNVPCouplingBlock,
                       {'subnet_constructor': F_conv, 'clamp': 1.0},
                       conditions=c1,
                       name='conv::c1')
        flatten = Ff.Node(conv,
                          Fm.Flatten,
                          {},
                          name='flatten')

        linear = Ff.Node(flatten,
                         Fm.RNVPCouplingBlock,
                         {'subnet_constructor': F_fully_connected, 'clamp': 1.0},
                         conditions=[c2, c3],
                         name='linear::c2|c3')

        outp = Ff.OutputNode(linear, name='output')
        self.test_net = Ff.GraphINN([inp, c1, conv, flatten, c2, c3, linear, outp])

    def test_output_shape(self):

        if self.skip_all:
            raise unittest.SkipTest("No CUDA-device found, skipping CUDA test.")

        y = self.test_net(self.x, c=[self.c1, self.c2, self.c3], jac=False)[0]
        self.assertTrue(isinstance(y, type(self.x)), f"{type(y)}")

        exp = torch.Size([self.batch_size, self.inp_size[0] * self.inp_size[1] * self.inp_size[2]])
        self.assertEqual(y.shape, exp, f"{y.shape}")

        # Assert that wrong condition inputs throw exceptions
        with self.assertRaises(Exception) as context:
            y = self.test_net(self.x, c=[self.c2, self.c1, self.c3])

        c2a = torch.randn(self.batch_size, self.c2_size[0] + 4, *self.c2_size[1:]).to(self.c2.device)
        with self.assertRaises(Exception) as context:
            y = self.test_net(self.x, c=[self.c1, c2a, self.c3])

        c1a = torch.randn(self.batch_size, *self.c1_size[:2], self.c1_size[2] + 1).to(self.c1.device)
        with self.assertRaises(Exception) as context:
            y = self.test_net(self.x, c=[c1a, self.c2, self.c3])

    def test_inverse(self):

        if self.skip_all:
            raise unittest.SkipTest("No CUDA-device found, skipping CUDA test.")

        y, j = self.test_net(self.x, c=[self.c1, self.c2, self.c3])
        x_re, j_re = self.test_net(y, c=[self.c1, self.c2, self.c3], rev=True)

        obs = torch.max(torch.abs(self.x - x_re))
        obs_j = torch.max(torch.abs(j + j_re))
        self.assertTrue(obs < self.inv_tol, f"Inversion {obs} !< {self.inv_tol}")
        self.assertTrue(obs_j  < self.inv_tol, f"Jacobian inversion {obs} !< {self.inv_tol}")

    def test_jacobian(self):

        if self.skip_all:
            raise unittest.SkipTest("No CUDA-device found, skipping CUDA test.")

        # Compute log det of Jacobian
        logdet = self.test_net(self.x, c=[self.c1, self.c2, self.c3])[1]
        # Approximate log det of Jacobian numerically
        logdet_num = self.test_net.log_jacobian_numerical(self.x, c=[self.c1, self.c2, self.c3], h=1e-3)
        # Check that they are the same (within tolerance)
        obs = torch.allclose(logdet, logdet_num, atol=np.inf, rtol=0.03)
        self.assertTrue(obs, f"Numerical Jacobian check {logdet, logdet_num}")


class ConditioningTestCuda(ConditioningTest):

    def __init__(self, *args):
        super().__init__(*args)

        if torch.cuda.is_available():
            self.x = self.x.cuda()
            self.c1 = self.c1.cuda()
            self.c2 = self.c2.cuda()
            self.c3 = self.c3.cuda()
            self.test_net.cuda()
        else:
            self.skip_all = True


if __name__ == '__main__':
    unittest.main()
