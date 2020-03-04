import torch
import torch.nn as nn
import numpy as np


def orth_correction(R):
    R[0] /= torch.norm(R[0])
    for i in range(1, R.shape[0]):

        R[i] -= torch.sum( R[:i].t() * torch.matmul(R[:i], R[i]), dim=1)
        R[i] /= torch.norm(R[i])

def correct_weights(module, grad_in, grad_out):

    module.back_counter += 1

    if module.back_counter > module.correction_interval:
        module.back_counter = np.random.randint(0, module.correction_interval) // 4
        orth_correction(module.weights.data)

class OrthogonalTransform(nn.Module):
    '''  '''

    def __init__(self, dims_in, correction_interval=256, clamp=5.):
        super().__init__()
        self.width = dims_in[0][0]
        self.clamp = clamp

        self.correction_interval = correction_interval
        self.back_counter = np.random.randint(0, correction_interval) // 2

        self.weights = torch.randn(self.width, self.width)
        self.weights = self.weights + self.weights.t()
        self.weights, S, V = torch.svd(self.weights)

        self.weights = nn.Parameter(self.weights)

        self.bias = nn.Parameter(0.05 * torch.randn(self.width))
        self.scaling = nn.Parameter(0.02 * torch.randn(self.width))

        self.register_backward_hook(correct_weights)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s/self.clamp))

    def log_e(self, s):
        '''log of the nonlinear function e'''
        return self.clamp * 0.636 * torch.atan(s/self.clamp)

    def forward(self, x, rev=False):
        if rev:
            return [(x[0] / self.e(self.scaling) - self.bias).mm(self.weights.t())]
        return [(x[0].mm(self.weights) + self.bias) * self.e(self.scaling)]

    def jacobian(self, x, rev=False):
        return torch.sum(self.log_e(self.scaling)).view(1,).expand(x[0].shape[0])

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class HouseholderPerm(nn.Module):

    def __init__(self, dims_in, dims_c=[], n_reflections=1, fixed=False):
        super().__init__()
        self.width = dims_in[0][0]
        self.n_reflections = n_reflections
        self.fixed = fixed
        self.conditional = (len(dims_c) > 0)

        if self.conditional:
            assert len(dims_c) == 1, "No more than one conditional input supported."
            assert not self.fixed, "Permutation can't be fixed and conditional simultaneously."
            assert prod(dims_c[0]) == self.width * self.n_reflections,\
                "Dimensions of input, n_reflections and condition don't agree."
        else:
            if self.fixed:
                # init randomly
                init = torch.randn(self.width, self.n_reflections)
            else:
                # init close to identity
                init = torch.eye(self.width, self.n_reflections)
                init += torch.randn_like(init) * 0.1
            Vs = torch.unbind(init, dim=-1)
            self.Vs = [nn.Parameter(V) for V in Vs]
            for i, V in enumerate(self.Vs):
                V.requires_grad = not self.fixed
                self.register_parameter(f'V_{i}', V)

        if self.fixed:
            I = torch.eye(self.width)
            self.W = I - 2 * torch.ger(self.Vs[0], self.Vs[0]) / torch.dot(self.Vs[0], self.Vs[0])
            for i in range(1, self.n_reflections):
                self.W = self.W.mm(I - 2 * torch.ger(self.Vs[i], self.Vs[i]) / torch.dot(self.Vs[i], self.Vs[i]))
            self.W = nn.Parameter(self.W, requires_grad=False)
            self.register_parameter('weight', self.W)

    def forward(self, x, c=[], rev=False):

        if self.conditional:
            Vs = torch.unbind(c[0].reshape(-1, self.width, self.n_reflections), dim=-1)

            xW = x[0]
            for i in range(self.n_reflections):
                if not rev:
                    V = Vs[i]
                else:
                    V = Vs[-i - 1]
                VVt = torch.matmul(V.unsqueeze(-1), V.unsqueeze(-2))
                VtV = torch.matmul(V.unsqueeze(-2), V.unsqueeze(-1)).squeeze(-1)
                xW = xW - torch.matmul(xW.unsqueeze(-2), VVt).squeeze() * (2/VtV)
            return [xW]

        else:
            if self.fixed:
                W = self.W
            else:
                I = torch.eye(self.width, device=x[0].device)
                W = I - 2 * torch.ger(self.Vs[0], self.Vs[0]) / torch.dot(self.Vs[0], self.Vs[0])
                for i in range(1, self.n_reflections):
                    W = W.mm(I - 2 * torch.ger(self.Vs[i], self.Vs[i]) / torch.dot(self.Vs[i], self.Vs[i]))

            if not rev:
                return [x[0].mm(W)]
            else:
                return [x[0].mm(W.t())]


    def jacobian(self, x, rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


import warnings

def _deprecated_by(orig_class):
    class deprecated_class(orig_class):
        def __init__(self, *args, **kwargs):

            warnings.warn(F"{self.__class__.__name__} is deprecated and will be removed in the public release. "
                          F"Use {orig_class.__name__} instead.",
                          DeprecationWarning)
            super().__init__(*args, **kwargs)

    return deprecated_class

orthogonal_layer = _deprecated_by(OrthogonalTransform)
