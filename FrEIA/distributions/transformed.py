import torch
from torch.distributions import Distribution

from FrEIA.modules import InvertibleModule
from FrEIA.modules.inverse import Inverse


class PushForwardDistribution(Distribution):
    # todo conditioning!
    def __init__(self, base_distribution: Distribution,
                 transform: InvertibleModule):
        super().__init__(transform.output_dims(transform.dims_in))
        self.base_distribution = base_distribution
        self.transform = transform

    def rsample(self, sample_shape=torch.Size):
        base_samples = self.base_distribution.rsample(sample_shape)
        # For now, only SequenceINN and GraphINN take
        # non-tuples as input and return non-tuples
        tuple_convert = (
                not hasattr(self.transform, "force_tuple_output")
                or self.transform.force_tuple_output
        )
        if tuple_convert:
            base_samples = (base_samples,)

        samples, _ = self.transform(base_samples, jac=False)

        if tuple_convert:
            samples = samples[0]
        return samples

    def log_prob(self, value: torch.Tensor):
        latent, log_abs_det = self.transform(value, jac=True, rev=True)
        return self.base_distribution.log_prob(latent) + log_abs_det


class PullBackDistribution(PushForwardDistribution):
    def __init__(self, base_distribution: Distribution,
                 transform: InvertibleModule):
        super().__init__(base_distribution, Inverse(transform))
