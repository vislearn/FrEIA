
import torch


class Distribution:
    def sample(self, size: torch.Size = (), temperature: float = 1.0) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
