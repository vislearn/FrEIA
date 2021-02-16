import unittest

import torch
import torch.nn as nn
import torch.optim

from FrEIA.modules import *
from FrEIA.framework import *

inp_size = 100
VERBOSE = False

inp = InputNode(inp_size, name='input')
orthog_layer = Node([inp.out0], OrthogonalTransform, {'correction_interval':100})
permute_1 = Node([orthog_layer.out0], PermuteRandom, {'seed':0})
permute_2 = Node([permute_1.out0], HouseholderPerm, {'n_reflections':10})
outp= OutputNode([permute_2.out0], name='output')

test_net = GraphINN([inp, orthog_layer, permute_1, permute_2, outp])

optim = torch.optim.SGD(test_net.parameters(), lr=5e-1)

#for name, p in test_net.named_parameters():
    #print(name, p.shape, p.requires_grad)


class OrthogonalTest(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)

        self.batch_size = 256
        self.tol = 1e-4
        torch.manual_seed(0)

    def test_inverse(self):
        return
        x = torch.randn(self.batch_size, inp_size)

        y = test_net(x)
        x_re = test_net(y, rev=True)

        if torch.max(torch.abs(x - x_re)) > self.tol:
            print(torch.max(torch.abs(x - x_re)).item(), end='   ')
            print(torch.mean(torch.abs(x - x_re)).item())

        self.assertTrue(torch.max(torch.abs(x - x_re)) < self.tol)

    def test_param_update(self):
        # TODO: there should be some quantifyable test condition at least...

        for i in range(2500):
            optim.zero_grad()

            x = torch.randn(self.batch_size, inp_size)
            y = test_net(x, jac=False)[0]

            loss = torch.mean((y-x)**2)
            loss.backward()

            for name, p in test_net.named_parameters():
                if 'weights' in name:
                    gp = torch.mm(p.grad, p.data.t())
                    p.grad = torch.mm(gp - gp.t(), p.data)

                    weights = p.data

            optim.step()

            if i%25 == 0:
                WWt = torch.mm(weights, weights.t())
                WWt -= torch.eye(weights.shape[0])
                if VERBOSE:
                    print(loss.item(), end='\t')
                    print(torch.max(torch.abs(WWt)).item(), end='\t')
                    print(torch.mean(WWt**2).item(), end='\t')
                    print()

    def test_cuda(self):
        return

        test_net.to('cuda')
        x = torch.randn(self.batch_size, inp_size).cuda()

        y = test_net(x, jac=False)[0]
        x_re = test_net(y, rev=True, jac=False)[0]

        self.assertTrue(torch.max(torch.abs(x - x_re)) < self.tol)
        test_net.to('cpu')


if __name__ == '__main__':
    unittest.main()
