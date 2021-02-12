from . import InvertibleModule

import torch
import torch.nn as nn
import numpy as np

def _fast_h(v, stride=2):
    """
    Fast product of a series of Householder matrices. This implementation is oriented to the one introducesd in:
    https://invertibleworkshop.github.io/accepted_papers/pdfs/10.pdf
    This makes use of method 2 in: https://ecommons.cornell.edu/bitstream/handle/1813/6521/85-681.pdf?sequence=1&isAllowed=y

    :param v: Batched series of Householder matrices. The last dim is the dim of one vector and the second last is the
    number of elements in one product. This is the min amount of dims that need to be present.
    All further ones are considered batch dimensions.
    :param stride: Controls the number of parallel operations by the WY representation (see paper)
    should not be larger than half the number of matrices in one product.
    :return: The batched product of Householder matrices defined by v
    """
    assert v.ndim > 1
    assert stride <= v.shape[-2]

    d, m = v.shape[-2], v.shape[-1]
    k = d // stride
    last = k * stride
    v = v / torch.norm(v, dim=-1, p=2, keepdim=True)
    v = v.unsqueeze(-1)
    u = 2 * v
    ID = torch.eye(m, device=u.device)
    for dim in range(v.ndim-3):
        ID = ID.unsqueeze(0)

    # step 1 (compute intermediate groupings P_i)
    W = u[..., 0:last:stride, :, :]
    Y = v[..., 0:last:stride, :, :]

    for idx in range(1, stride):
        Pt = ID - torch.matmul(u[..., idx:last:stride, :, :], v[..., idx:last:stride, :, :].transpose(-1, -2))
        W = torch.cat([W, u[..., idx:last:stride, :, :]], dim=-1)
        Y = torch.cat([torch.matmul(Pt, Y), v[..., idx:last:stride, :, :]], dim=-1)

    # step 2 (multiply the WY reps)
    P = ID - torch.matmul(W[..., k-1, :, :], Y[..., k-1, :, :].transpose(-1, -2))
    for idx in reversed(range(0, k-1)):
        P = P - torch.matmul(W[..., idx, :, :], torch.matmul(Y[..., idx, :, :].transpose(-1, -2), P))

    # deal with the residual, using a stride of 2 here maxes the amount of parallel ops
    if d > last:
        even_end = d if (d-last) % 2 == 0 else d - 1
        W_resi = u[..., last:even_end:2, :, :]
        Y_resi = v[..., last:even_end:2, :, :]
        for idx in range(last+1, d if d == last+1 else last+2):
            Pt = ID - torch.matmul(u[..., idx:even_end:2, :, :], v[..., idx:even_end:2, :, :].transpose(-1, -2))
            W_resi = torch.cat([W_resi, u[..., idx:even_end:2, :, :]], dim=-1)
            Y_resi = torch.cat([torch.matmul(Pt, Y_resi), v[..., idx:even_end:2, :, :]], dim=-1)

        for idx in range(0, W_resi.shape[-3]):
            P = P - torch.matmul(P, torch.matmul(W_resi[..., idx, :, :], Y_resi[..., idx, :, :].transpose(-1, -2)))

        if even_end != d:
            P = P - torch.matmul(P, torch.matmul(u[..., -1, :, :], v[..., -1, :, :].transpose(-1, -2)))

    return P

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

class OrthogonalTransform(InvertibleModule):
    '''  '''

    def __init__(self, dims_in, dims_c=None, correction_interval=256, clamp=5.):
        super().__init__(dims_in, dims_c)
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

    def forward(self, x, rev=False, jac=True):
        j = torch.sum(self.log_e(self.scaling)).view(1,).expand(x[0].shape[0])
        if rev:
            return [(x[0] / self.e(self.scaling) - self.bias).mm(self.weights.t())], -j
        return [(x[0].mm(self.weights) + self.bias) * self.e(self.scaling)], j

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class HouseholderPerm(InvertibleModule):

    def __init__(self, dims_in, dims_c=[], n_reflections=1, fixed=False):
        super().__init__(dims_in, dims_c)
        self.width = dims_in[0][0]
        self.n_reflections = n_reflections
        self.fixed = fixed
        self.conditional = (len(dims_c) > 0)

        if self.conditional:
            assert len(dims_c) == 1, "No more than one conditional input supported."
            assert not self.fixed, "Permutation can't be fixed and conditional simultaneously."
            assert np.prod(dims_c[0]) == self.width * self.n_reflections,\
                "Dimensions of input, n_reflections and condition don't agree."
        else:
            if self.fixed:
                # init randomly
                init = torch.randn(self.width, self.n_reflections)
            else:
                # init close to identity
                init = torch.eye(self.width, self.n_reflections)
                init += torch.randn_like(init) * 0.1
            Vs = init.transpose(-1, -2)
            self.Vs = nn.Parameter(Vs)

            Vs.requires_grad = not self.fixed
            self.register_parameter('Vs', self.Vs)

        if self.fixed:
            self.W = _fast_h(self.Vs)
            self.W = nn.Parameter(self.W, requires_grad=False)
            self.register_parameter('weight', self.W)

    def forward(self, x, c=[], rev=False, jac=True):

        if self.conditional:
            Vs = c[0].reshape(-1, self.width, self.n_reflections).transpose(-1, -2)
            W = _fast_h(Vs)
        else:
            if self.fixed:
                W = self.W
            else:
                W = _fast_h(self.Vs)

        if not rev:
            return [x[0].mm(W)], 0.
        else:
            return [x[0].mm(W.transpose(-1, -2))], 0.

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims
