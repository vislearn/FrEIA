import torch
import torch.nn as nn
from torch.nn.functional import conv2d, conv_transpose2d

import numpy as np



class ActNorm(nn.Module):

    def __init__(self, dims_in, init_data=None):
        super().__init__()
        self.dims_in = dims_in[0]
        param_dims = [1, self.dims_in[0]] + [1 for i in range(len(self.dims_in) - 1)]
        self.scale = nn.Parameter(torch.zeros(*param_dims))
        self.bias = nn.Parameter(torch.zeros(*param_dims))

        if init_data:
            self.initialize_with_data(init_data)
        else:
            self.init_on_next_batch = True

        def on_load_state_dict(*args):
            # when this module is loading state dict, we SHOULDN'T init with data,
            # because that will reset the trained parameters. Registering a hook
            # that disable this initialisation.
            self.init_on_next_batch = False
        self._register_load_state_dict_pre_hook(on_load_state_dict)

    def initialize_with_data(self, data):
        # Initialize to mean 0 and std 1 with sample batch
        # 'data' expected to be of shape (batch, channels[, ...])
        assert all([data.shape[i+1] == self.dims_in[i] for i in range(len(self.dims_in))]),\
            "Can't initialize ActNorm layer, provided data don't match input dimensions."
        self.scale.data.view(-1)[:] \
            = torch.log(1 / data.transpose(0,1).contiguous().view(self.dims_in[0], -1).std(dim=-1))
        data = data * self.scale.exp()
        self.bias.data.view(-1)[:] \
            = -data.transpose(0,1).contiguous().view(self.dims_in[0], -1).mean(dim=-1)
        self.init_on_next_batch = False

    def forward(self, x, rev=False):
        if self.init_on_next_batch:
            self.initialize_with_data(x[0])

        if not rev:
            return [x[0] * self.scale.exp() + self.bias]
        else:
            return [(x[0] - self.bias) / self.scale.exp()]

    def jacobian(self, x, rev=False):
        if not rev:
            return (self.scale.sum() * np.prod(self.dims_in[1:])).repeat(x[0].shape[0])
        else:
            return -self.jacobian(x)

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims



class IResNetLayer(nn.Module):
    """
    Implementation of the i-ResNet architecture as proposed in
    https://arxiv.org/pdf/1811.00995.pdf
    """

    def __init__(self, dims_in, dims_c=[],
                 internal_size=None,
                 n_internal_layers=1,
                 jacobian_iterations=20,
                 hutchinson_samples=1,
                 fixed_point_iterations=50,
                 lipschitz_iterations=10,
                 lipschitz_batchsize=10,
                 spectral_norm_max=0.8):
        super().__init__()

        if internal_size:
            self.internal_size = internal_size
        else:
            self.internal_size = 2 * dims_in[0][0]
        self.n_internal_layers = n_internal_layers
        self.jacobian_iterations = jacobian_iterations
        self.hutchinson_samples = hutchinson_samples
        self.fixed_point_iterations = fixed_point_iterations
        self.lipschitz_iterations = lipschitz_iterations
        self.lipschitz_batchsize = lipschitz_batchsize
        self.spectral_norm_max = spectral_norm_max
        assert 0 < spectral_norm_max <= 1, "spectral_norm_max must be in (0,1]."

        self.dims_in = dims_in[0]
        if len(self.dims_in) == 1:
            # Linear case
            self.layers = [nn.Linear(self.dims_in[0], self.internal_size),]
            for i in range(self.n_internal_layers):
                self.layers.append(nn.Linear(self.internal_size, self.internal_size))
            self.layers.append(nn.Linear(self.internal_size, self.dims_in[0]))
        else:
            # Convolutional case
            self.layers = [nn.Conv2d(self.dims_in[0], self.internal_size, 3, padding=1),]
            for i in range(self.n_internal_layers):
                self.layers.append(nn.Conv2d(self.internal_size, self.internal_size, 3, padding=1))
            self.layers.append(nn.Conv2d(self.internal_size, self.dims_in[0], 3, padding=1))
        elus = [nn.ELU() for i in range(len(self.layers))]
        module_list = sum(zip(self.layers, elus), ())[:-1] # interleaves the lists
        self.residual = nn.Sequential(*module_list)


    def lipschitz_correction(self):
        with torch.no_grad():
            # Power method to approximate spectral norm
            # Following https://arxiv.org/pdf/1804.04368.pdf
            for i in range(len(self.layers)):
                W = self.layers[i].weight
                x = torch.randn(self.lipschitz_batchsize, W.shape[1], *self.dims_in[1:], device=W.device)

                if len(self.dims_in) == 1:
                    # Linear case
                    for j in range(self.lipschitz_iterations):
                        x = W.t().matmul(W.matmul(x.unsqueeze(-1))).squeeze(-1)
                    spectral_norm = (torch.norm(W.matmul(x.unsqueeze(-1)).squeeze(-1), dim=1) /\
                                     torch.norm(x, dim=1)).max()
                else:
                    # Convolutional case
                    for j in range(self.lipschitz_iterations):
                        x = conv2d(x, W)
                        x = conv_transpose2d(x, W)
                    spectral_norm = (torch.norm(conv2d(x, W).view(self.lipschitz_batchsize, -1), dim=1) /\
                                     torch.norm(x.view(self.lipschitz_batchsize, -1), dim=1)).max()

                if spectral_norm > self.spectral_norm_max:
                    self.layers[i].weight.data *= self.spectral_norm_max / spectral_norm


    def forward(self, x, c=[], rev=False):
        if not rev:
            return [x[0] + self.residual(x[0])]
        else:
            # Fixed-point iteration (works if residual has Lipschitz constant < 1)
            y = x[0]
            with torch.no_grad():
                x_hat = x[0]
                for i in range(self.fixed_point_iterations):
                    x_hat = y - self.residual(x_hat)
            return [y - self.residual(x_hat.detach())]


    def jacobian(self, x, c=[], rev=False):
        if rev:
            return -self.jacobian(x, c=c)

        # Initialize log determinant of Jacobian to zero
        batch_size = x[0].shape[0]
        logdet_J = x[0].new_zeros(batch_size)
        # Make sure we can get vector-Jacobian product w.r.t. x even if x is the network input
        if x[0].is_leaf:
            x[0].requires_grad = True

        # Sample random vectors for Hutchinson trace estimate
        v_right = [torch.randn_like(x[0]).sign() for i in range(self.hutchinson_samples)]
        v_left = [v.clone() for v in v_right]

        # Compute terms of power series
        for k in range(1, self.jacobian_iterations+1):
            # Estimate trace of Jacobian of residual branch
            trace_est = []
            for i in range(self.hutchinson_samples):
                # Compute vector-Jacobian product v.t() * J
                residual = self.residual(x[0])
                v_left[i] = torch.autograd.grad(outputs=[residual],
                                                inputs=x,
                                                grad_outputs=[v_left[i]])[0]
                trace_est.append(v_left[i].view(batch_size, 1, -1).matmul(v_right[i].view(batch_size, -1, 1)).squeeze(-1).squeeze(-1))
            if len(trace_est) > 1:
                trace_est = torch.stack(trace_est).mean(dim=0)
            else:
                trace_est = trace_est[0]
            # Update power series approximation of log determinant for the whole block
            logdet_J = logdet_J + (-1)**(k+1) * trace_est / k

        # # Shorter version when self.hutchinson_samples is fixed to one
        # v_right = torch.randn_like(x[0])
        # v_left = v_right.clone()
        # residual = self.residual(x[0])
        # for k in range(1, self.jacobian_iterations+1):
        #     # Compute vector-Jacobian product v.t() * J
        #     v_left = torch.autograd.grad(outputs=[residual],
        #                                  inputs=x,
        #                                  grad_outputs=[v_left],
        #                                  retain_graph=(k < self.jacobian_iterations))[0]
        #     # Iterate power series approximation of log determinant
        #     trace_est = v_left.view(batch_size, 1, -1).matmul(v_right.view(batch_size, -1, 1)).squeeze(-1).squeeze(-1)
        #     logdet_J = logdet_J + (-1)**(k+1) * trace_est / k

        return logdet_J


    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims
