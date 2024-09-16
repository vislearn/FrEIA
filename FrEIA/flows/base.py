
from freia.core import Invertible

class Flow(Invertible):
    def __init__(self, transform, distribution):
        self.transform = transform
        self.distribution = distribution

    def forward(self, x):
        z, logdet = self.transform.forward(x)

        logp = self.distribution.log_prob(z)

        nll = -(logp + logdet)

        return z, nll

    def sample_transform(self, size, temperature):
        z = self.distribution.sample(size, temperature)

        x, _ = self.transform.inverse(z)

        return x


class RecurrentFlow(Flow):
    def forward(self, x):
        z = x
        logdet = None
        for t in range(...):
            z, logdet = self.transform.forward(z, t)