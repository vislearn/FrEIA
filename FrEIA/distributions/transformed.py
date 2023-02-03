from typing import List

import torch
from torch.distributions import Distribution

from FrEIA.modules import InvertibleModule
from FrEIA.modules.inverse import Inverse
from FrEIA.utils import force_to, output_dims_compatible


class PushForwardDistribution(Distribution):
    arg_constraints = {}

    def __init__(self, base_distribution: Distribution,
                 transform: InvertibleModule):
        # Hack as SequenceINN and GraphINN do not work with input/output shape API
        event_shape = output_dims_compatible(transform)
        super().__init__(torch.Size(), event_shape)
        self.base_distribution = base_distribution
        self.transform = transform

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size = torch.Size(), conditions: List[torch.Tensor] = None) -> torch.Tensor:
        base_samples = self.base_distribution.sample(sample_shape)

        # For now, only SequenceINN and GraphINN take
        # non-tuples as input and return non-tuples
        tuple_convert = (
                not hasattr(self.transform, "force_tuple_output")
                or self.transform.force_tuple_output
        )
        if tuple_convert:
            base_samples = (base_samples,)

        kwargs = dict()
        if conditions is not None:
            kwargs["c"] = conditions
        samples, _ = self.transform(base_samples, jac=False, **kwargs)

        if tuple_convert:
            samples = samples[0]
        return samples

    def rsample(self, sample_shape: torch.Size = torch.Size(), conditions: List[torch.Tensor] = None) -> torch.Tensor:
        try:
            base_samples = self.base_distribution.rsample(sample_shape)
        except NotImplementedError:
            base_samples = self.base_distribution.sample(sample_shape)
        # For now, only SequenceINN and GraphINN take
        # non-tuples as input and return non-tuples
        tuple_convert = (
                not hasattr(self.transform, "force_tuple_output")
                or self.transform.force_tuple_output
        )
        if tuple_convert:
            base_samples = (base_samples,)

        kwargs = dict()
        if conditions is not None:
            kwargs["c"] = conditions
        samples, _ = self.transform(base_samples, jac=False, **kwargs)

        if tuple_convert:
            samples = samples[0]
        return samples

    def log_prob(self, value: torch.Tensor, conditions: List[torch.Tensor] = None):
        # expected_shape = self.transform.output_dims(self.transform.dims_in)[0]
        # data_shape = value.shape[-len(expected_shape):]
        # if data_shape != expected_shape:
        # raise ValueError(f"Got input of trailing shape {data_shape}, but expected {expected_shape}.")

        # For now, only SequenceINN and GraphINN take
        # non-tuples as input and return non-tuples
        tuple_convert = (
                not hasattr(self.transform, "force_tuple_output")
                or self.transform.force_tuple_output
        )
        if tuple_convert:
            value = (value,)
        kwargs = dict()
        if conditions is not None:
            kwargs["c"] = conditions
        latent, log_abs_det = self.transform(value, **kwargs, jac=False, rev=True)
        if tuple_convert:
            latent = latent[0]
        return self.base_distribution.log_prob(latent) + log_abs_det

    def force_to(self, *args, **kwargs):
        force_to(self.base_distribution, *args, **kwargs)
        self.transform = self.transform.to(*args, **kwargs)


class PullBackDistribution(PushForwardDistribution):
    def __init__(self, base_distribution: Distribution,
                 transform: InvertibleModule):
        super().__init__(base_distribution, Inverse(transform))
