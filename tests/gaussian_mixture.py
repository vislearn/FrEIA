import unittest

import torch
import torch.nn as nn

from FrEIA.modules import *
from FrEIA.framework import *


batch_size = 5
n_components = 4
n_dims = 12

x_size = (n_dims,)
w_size = (n_components,)
mu_size = (n_components, n_dims)
U_size = (n_components, n_dims * (n_dims + 1) // 2)
i_size = (0,)

inp = InputNode(*x_size, name='input')
w = ConditionNode(*w_size, name='w')
mu = ConditionNode(*mu_size, name='mu')
U = ConditionNode(*U_size, name='U')
i = ConditionNode(*i_size, name='U')
gmm = Node(inp,
           GaussianMixtureModel,
           {},
           conditions=[w, mu, U, i],
           name='gmm')
out = OutputNode(gmm, name='output')
test_net = ReversibleGraphNet([inp, w, mu, U, i, gmm, out], verbose=False)


class GMMTest(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)
        torch.manual_seed(0)
        self.tol = 2e-3

    def test_inverse_fixed_components(self):
        x = torch.randn(batch_size, *x_size)
        w = torch.randn(batch_size, *w_size)
        w = GaussianMixtureModel.normalize_weights(w)
        mu = torch.randn(batch_size, *mu_size)
        U = torch.randn(batch_size, *U_size)
        i = torch.randint(0, n_components, (batch_size,))

        z = test_net(x, c=[w, mu, U, i])
        x_re = test_net(z, c=[w, mu, U, i], rev=True)

        if torch.max(torch.abs(x - x_re)) > self.tol:
            print(torch.max(torch.abs(x - x_re)).item(), end='   ')
            print(torch.mean(torch.abs(x - x_re)).item())
        self.assertTrue(torch.max(torch.abs(x - x_re)) < self.tol)


        z = test_net(x, c=[w, mu, U, 12345])
        x_re = test_net(z, c=[w, mu, U, 12345], rev=True)

        if torch.max(torch.abs(x - x_re)) > self.tol:
            print(torch.max(torch.abs(x - x_re)).item(), end='   ')
            print(torch.mean(torch.abs(x - x_re)).item())
        self.assertTrue(torch.max(torch.abs(x - x_re)) < self.tol)


    def test_inverse_all_components(self):
        x = torch.randn(batch_size, *x_size)
        w = torch.randn(batch_size, *w_size)
        w = GaussianMixtureModel.normalize_weights(w)
        mu = torch.randn(batch_size, *mu_size)
        U = torch.randn(batch_size, *U_size)

        z = test_net(x, c=[w, mu, U, None])
        x_re = test_net(z, c=[w, mu, U, None], rev=True)

        for comp_idx in range(n_components):
            if torch.max(torch.abs(x - x_re[:,comp_idx,:])) > self.tol:
                print(torch.max(torch.abs(x - x_re[:,comp_idx,:])).item(), end='   ')
                print(torch.mean(torch.abs(x - x_re[:,comp_idx,:])).item())
            self.assertTrue(torch.max(torch.abs(x - x_re[:,comp_idx,:])) < self.tol)

        # Check that nll losses don't throw errors
        jac = test_net.log_jacobian(x, c=[w, mu, U, None])
        nll = GaussianMixtureModel.nll_loss(w, z, jac)
        nll_bound = GaussianMixtureModel.nll_upper_bound(w, z, jac)
        # print(nll, nll_bound)


    def test_jacobian_fixed_components(self):
        x = torch.randn(batch_size, *x_size)
        w = torch.randn(batch_size, *w_size)
        w = GaussianMixtureModel.normalize_weights(w)
        mu = torch.randn(batch_size, *mu_size)
        U = torch.randn(batch_size, *U_size)

        # Compute log det of Jacobian
        z = test_net(x, c=[w, mu, U, 12345])
        logdet = test_net.log_jacobian(x, c=[w, mu, U, 12345])
        # Approximate log det of Jacobian numerically
        logdet_num = test_net.log_jacobian_numerical(x, c=[w, mu, U, 12345])
        # Check that they are the same (within tolerance)
        self.assertTrue(torch.allclose(logdet, logdet_num, atol=0.01, rtol=0.01))


    # def test_all_components_cuda(self):
    #     dev = 'cuda'
    #     test_net.to(dev)

    #     x = torch.randn(batch_size, *x_size).to(dev)
    #     w = torch.randn(batch_size, *w_size).to(dev)
    #     mu = torch.randn(batch_size, *mu_size).to(dev)
    #     U = torch.randn(batch_size, *U_size).to(dev)

    #     z = test_net(x, c=[w, mu, U, None])
    #     x_re = test_net(z, c=[w, mu, U, None], rev=True)

    #     for comp_idx in range(n_components):
    #         if torch.max(torch.abs(x - x_re[:,comp_idx,:])) > self.tol:
    #             print(torch.max(torch.abs(x - x_re[:,comp_idx,:])).item(), end='   ')
    #             print(torch.mean(torch.abs(x - x_re[:,comp_idx,:])).item())
    #         self.assertTrue(torch.max(torch.abs(x - x_re[:,comp_idx,:])) < self.tol)

    #     test_net.to('cpu')


if __name__ == '__main__':
    unittest.main()
