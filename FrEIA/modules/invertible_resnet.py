import FrEIA.utils as utils
from . import InvertibleModule

import torch
import torch.nn as nn
from torch.nn.functional import conv2d, conv_transpose2d

import warnings


class ActNorm(InvertibleModule):
    """
    A technique to achieve stable flow initialization.

    First introduced in Kingma et al. 2018: https://arxiv.org/abs/1807.03039
    The module is similar to a traditional batch normalization layer, but the
    data mean and standard deviation are initialized from the first batch that
    is passed through the module. They are treated as learnable parameters from
    there on.

    Using ActNorm layers interspersed throughout an INN ensures that
    intermediate outputs of the INN have standard deviation 1 and mean 0, so
    that the training is stable at the start, avoiding exploding or zeroed
    outputs.
    Just as with standard batch normalization layers, ActNorm contains
    additional channel-wise scaling and bias parameters.
    """
    def __init__(self, dims_in, dims_c=None, init_data: torch.Tensor = None):
        super().__init__(dims_in, dims_c)

        self.register_buffer("is_initialized", torch.tensor(False))

        dim = next(iter(dims_in))[0]
        self.log_scale = nn.Parameter(torch.empty(1, dim))
        self.loc = nn.Parameter(torch.empty(1, dim))

        if init_data is not None:
            self.initialize(init_data)

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def initialize(self, batch: torch.Tensor):
        self.is_initialized.data = torch.tensor(True)
        self.log_scale.data = torch.log(torch.std(batch, dim=0, keepdim=True))
        self.loc.data = torch.mean(batch, dim=0, keepdim=True)

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use one input"
        return input_dims

    def forward(self, x, c=None, rev=False, jac=True):
        if c is not None:
            raise ValueError(f"{self.__class__.__name__} is unconditional.")

        x = x[0]

        if not self.is_initialized:
            self.initialize(x)

        if not rev:
            out = (x - self.loc) / self.scale
            log_jac_det = -utils.sum_except_batch(self.log_scale)
        else:
            out = self.scale * x + self.loc
            log_jac_det = utils.sum_except_batch(self.log_scale)

        return (out,), log_jac_det

    def load_state_dict(self, state_dict, strict=True):
        if list(state_dict.keys()) == ["scale", "bias"]:
            if strict:
                warnings.warn(DeprecationWarning(f"Parameter names in {self.__class__.__name__} have changed. "
                                                 f"Converting state_dict to new format. "
                                                 f"Please overwrite your old state_dicts."))

            # compatibiliy with old ActNorm
            state_dict = {
                "log_scale": -state_dict["scale"],
                "loc": -(torch.exp(-state_dict["scale"]) * state_dict["bias"]),
                "is_initialized": torch.tensor(True),
            }

        return super().load_state_dict(state_dict, strict)


class IResNetLayer(InvertibleModule):
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

        super().__init__(dims_in, dims_c)

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


    def forward(self, x, c=[], rev=False, jac=True):
        if jac:
            jac = self._jacobian(x, c, rev=rev)
        else:
            jac = None

        if not rev:
            return [x[0] + self.residual(x[0])], jac
        else:
            # Fixed-point iteration (works if residual has Lipschitz constant < 1)
            y = x[0]
            with torch.no_grad():
                x_hat = x[0]
                for i in range(self.fixed_point_iterations):
                    x_hat = y - self.residual(x_hat)
            return [y - self.residual(x_hat.detach())], jac


    def _jacobian(self, x, c=[], rev=False):
        if rev:
            return -self._jacobian(x, c=c)

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
