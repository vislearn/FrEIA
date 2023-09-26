import unittest

import torch.optim

import sys
sys.path.append('../')
from FrEIA.modules import *
from FrEIA.framework import *


class ActNormTest(unittest.TestCase):
    batch_size = 256

    def test_linear(self):
        torch.manual_seed(0)
        inp_size_linear = (20,)
        
        act_norm = ActNorm([inp_size_linear])
        
        x = torch.randn(self.batch_size, *inp_size_linear)
        x = x * torch.rand_like(x) + torch.randn_like(x)
        y, = act_norm([x], jac=False)[0]
        self.assertStandardMoments(y)

    def test_conv(self):
        torch.manual_seed(0)
        
        inp_size_conv = (3, 10, 10)
        act_norm = ActNorm([inp_size_conv])

        x = torch.randn(self.batch_size, *inp_size_conv)
        x = x * torch.rand_like(x) + torch.randn_like(x)
        y, = act_norm([x], jac=False)[0]

        y_ = y.reshape(self.batch_size, inp_size_conv[0], -1)
        self.assertStandardMoments(y_)

    def assertStandardMoments(self, data):
        dims = [0] + list(range(2, data.ndim))
        mean = torch.mean(data, dim=dims)
        std = torch.std(data, dim=dims)

        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-7))
        self.assertTrue(torch.allclose(std,  torch.ones_like(std)))


class IResNetTest(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)

        self.batch_size = 7
        self.inp_size_linear = (20,)
        self.inp_size_conv = (3, 10, 10)
        self.tol = 1e-6
        torch.manual_seed(0)

        nodes = [InputNode(*self.inp_size_linear, name='input')]
        cond = ConditionNode(*self.inp_size_linear, name='cond')
        for i in range(5):
            nodes.append(Node(nodes[-1], ActNorm, {},
                              name=f'actnorm_{i}'))
            nodes.append(Node(nodes[-1], IResNetLayer,
                              {'hutchinson_samples': 20,
                               'internal_size': 100,
                               'n_internal_layers': 3},
                              conditions=[cond],
                              name=f'i_resnet_{i}'))
        nodes.append(OutputNode(nodes[-1], name='output'))
        self.i_resnet_linear = GraphINN(nodes + [cond,], verbose=False)

        for node in self.i_resnet_linear.node_list:
            if isinstance(node.module, IResNetLayer):
                node.module.lipschitz_correction()


        nodes = [InputNode(*self.inp_size_conv, name='input')]
        for i in range(5):
            nodes.append(Node(nodes[-1], ActNorm, {},
                              name=f'actnorm_{i}'))
            nodes.append(Node(nodes[-1], IResNetLayer, {'hutchinson_samples': 20},
                              name=f'i_resnet_{i}'))
        nodes.append(OutputNode(nodes[-1], name='output'))
        self.i_resnet_conv = GraphINN(nodes, verbose=False)

        for node in self.i_resnet_conv.node_list:
            if isinstance(node.module, IResNetLayer):
                node.module.lipschitz_correction()


    def test_inverse(self):
        x = torch.randn(self.batch_size, *self.inp_size_linear)
        x = x * torch.randn_like(x)
        x = x + torch.randn_like(x)
        c = torch.randn(self.batch_size, *self.inp_size_linear)

        y = self.i_resnet_linear(x, c, jac=False)[0]
        x_hat = self.i_resnet_linear(y, c, rev=True, jac=False)[0]
        # Check that inverse is close to input
        self.assertTrue(torch.allclose(x, x_hat, atol=self.tol))

        x = torch.randn(self.batch_size, *self.inp_size_conv)
        x = x * torch.randn_like(x)
        x = x + torch.randn_like(x)

        y = self.i_resnet_conv(x, jac=False)[0]
        x_hat = self.i_resnet_conv(y, rev=True, jac=False)[0]
        # Check that inverse is close to input
        self.assertTrue(torch.allclose(x, x_hat, atol=self.tol))


    def test_jacobian(self):
        x = torch.randn(self.batch_size, *self.inp_size_linear)
        x = x * torch.randn(self.batch_size, *[1 for i in range(len(self.inp_size_linear))])
        x = x + torch.randn(self.batch_size, *[1 for i in range(len(self.inp_size_linear))])
        c = torch.randn(self.batch_size, *self.inp_size_linear)

        # Estimate log det of Jacobian via power series
        z, logdet = self.i_resnet_linear(x, c=c)
        # Approximate log det of Jacobian numerically
        logdet_num = self.i_resnet_linear.log_jacobian_numerical(x, c=c)
        # Check that they are the same (with huge tolerance)
        # print(f'\n{logdet}\n{logdet_num}')
        self.assertTrue(torch.allclose(logdet, logdet_num, atol=1.5, rtol=0.15))


        x = torch.randn(self.batch_size, *self.inp_size_conv)
        x = x * torch.randn(self.batch_size, *[1 for i in range(len(self.inp_size_conv))])
        x = x + torch.randn(self.batch_size, *[1 for i in range(len(self.inp_size_conv))])

        # Estimate log det of Jacobian via power series
        logdet = self.i_resnet_conv(x)[1]
        # Approximate log det of Jacobian numerically
        logdet_num = self.i_resnet_conv.log_jacobian_numerical(x)
        # Check that they are the same (with huge tolerance)
        # print(f'\n{logdet}\n{logdet_num}')
        self.assertTrue(torch.allclose(logdet, logdet_num, atol=1.5, rtol=0.1))


if __name__ == '__main__':
    unittest.main()
