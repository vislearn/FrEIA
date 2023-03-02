
import FrEIA.framework as ff
import FrEIA.modules as fm

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union


class SubnetFactory:
    def __init__(self, kind: str, widths: list, activation: Union[str, type(nn.Module)] = "relu", dropout: float = None, **kwargs):
        self.kind = kind
        self.kwargs = kwargs
        self.widths = widths
        self.activation = activation
        self.dropout = dropout

    def layer(self, dims_in, dims_out):
        kind = self.kind.lower()
        if kind == "dense":
                return nn.Linear(dims_in, dims_out, **self.kwargs)
        if kind == "conv":
            return nn.Conv2d(dims_in, dims_out, padding="same", **self.kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support layer kind {kind}.")

    def activation_layer(self, **kwargs):
        if isinstance(self.activation, nn.Module):
            return self.activation(**kwargs)
        if isinstance(self.activation, str):
            name = self.activation.lower()
            if name == "relu":
                return nn.ReLU(**kwargs)
            elif name == "elu":
                return nn.ELU(**kwargs)
            elif name == "selu":
                return nn.SELU(**kwargs)
            elif name == "leakyrelu":
                return nn.LeakyReLU(**kwargs)
            elif name == "tanh":
                return nn.Tanh(**kwargs)
            raise NotImplementedError(f"{self.__class__.__name__} does not support activation with name {name}.")
        raise NotImplementedError(f"{self.__class__.__name__} does not support activation type {self.activation}.")

    def dropout_layer(self, **kwargs):
        kind = self.kind.lower()
        if kind == "dense":
            return nn.Dropout1d(p=self.dropout, **kwargs)
        elif kind == "conv":
            return nn.Dropout2d(p=self.dropout, **kwargs)
        raise NotImplementedError(f"{self.__class__.__name__} does not support layer kind {kind}.")

    def __call__(self, dims_in, dims_out):
        network = nn.Sequential()

        network.add_module("Input_Layer", self.layer(dims_in, self.widths[0]))
        network.add_module("Input_Activation", self.activation_layer())

        for i in range(len(self.widths) - 1):
            if self.dropout is not None:
                network.add_module(f"Dropout_{i:02d}", self.dropout_layer())

            network.add_module(f"Hidden_Layer_{i:02d}", self.layer(self.widths[i], self.widths[i + 1]))
            network.add_module(f"Hidden_Activation_{i:02d}", self.activation_layer())

        network.add_module(f"Output Layer", self.layer(self.widths[-1], dims_out))

        network[-1].weight.data.zero_()
        network[-1].bias.data.zero_()

        return network


class TestUnconditionalCoupling:

    couplings = [
        fm.AllInOneBlock,
        fm.NICECouplingBlock,
        fm.RNVPCouplingBlock,
        fm.GLOWCouplingBlock,
        fm.GINCouplingBlock,
        fm.AffineCouplingOneSided,
        fm.RationalQuadraticSpline,
        fm.ElementwiseRationalQuadraticSpline,
        fm.LinearSpline,
    ]
    scenarios = [
        ((32, 3), "dense", [16, 32, 16], dict()),
        ((8, 3, 8, 8), "conv", [8, 8, 8], dict(kernel_size=1)),
        ((8, 3, 8, 8), "conv", [4, 6, 4], dict(kernel_size=3)),
    ]

    def test_forward_backward(self):
        for coupling_type in self.couplings:
            for batch_shape, network_kind, network_widths, network_kwargs in self.scenarios:
                subnet_constructor = SubnetFactory(kind=network_kind, widths=network_widths, **network_kwargs)
                inn = ff.SequenceINN(*batch_shape[1:])
                inn.append(coupling_type, subnet_constructor=subnet_constructor)

                sample_data = torch.randn(*batch_shape)

                latent, forward_logdet = inn(sample_data)
                assert latent.shape == sample_data.shape
                assert forward_logdet.dim() == 1

                reconstruction, backward_logdet = inn(latent, rev=True)
                assert reconstruction.shape == sample_data.shape
                assert backward_logdet.dim() == 1
                assert torch.allclose(sample_data, reconstruction, atol=1e-5,
                                      rtol=1e-3), f"MSE: {F.mse_loss(sample_data, reconstruction)}"
                assert torch.allclose(forward_logdet, -backward_logdet, atol=1e-5,
                                      rtol=1e-3), f"MSE: {F.mse_loss(forward_logdet, -backward_logdet)}"

class TestConditionalCoupling:

    couplings = [
        fm.AllInOneBlock,
        fm.NICECouplingBlock,
        fm.RNVPCouplingBlock,
        fm.GLOWCouplingBlock,
        fm.GINCouplingBlock,
        fm.AffineCouplingOneSided,
        fm.RationalQuadraticSpline,
        fm.ElementwiseRationalQuadraticSpline,
        fm.LinearSpline,
    ]
    scenarios = [
        ((32, 3), (32, 4), "dense", [16, 32, 16], dict()),
        ((8, 3, 8, 8), (8, 4, 8, 8), "conv", [8, 8, 8], dict(kernel_size=1)),
        ((8, 3, 8, 8), (8, 4, 8, 8), "conv", [4, 6, 4], dict(kernel_size=3)),
    ]

    def test_forward_backward(self):
        for coupling_type in self.couplings:
            for batch_shape, condition_shape, network_kind, network_widths, network_kwargs in self.scenarios:
                subnet_constructor = SubnetFactory(kind=network_kind, widths=network_widths, **network_kwargs)
                inn = ff.SequenceINN(*batch_shape[1:])
                inn.append(coupling_type, 
                        subnet_constructor=subnet_constructor, 
                        cond_shape=condition_shape[1:],
                        cond=0)

                sample_data = torch.randn(*batch_shape)
                sample_cond = torch.randn(*condition_shape)

                latent, forward_logdet = inn(sample_data, (sample_cond,))
                assert latent.shape == sample_data.shape
                assert forward_logdet.dim() == 1

                reconstruction, backward_logdet = inn(latent, (sample_cond,), rev=True)
                assert reconstruction.shape == sample_data.shape
                assert backward_logdet.dim() == 1

                assert torch.allclose(sample_data, reconstruction, atol=1e-5,
                                      rtol=1e-3), f"MSE: {F.mse_loss(sample_data, reconstruction)}"
                assert torch.allclose(forward_logdet, -backward_logdet, atol=1e-5,
                                      rtol=1e-3), f"MSE: {F.mse_loss(forward_logdet, -backward_logdet)}"
