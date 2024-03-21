
from abc import ABC
import torch.nn as nn

from typing import Any

from typing import TypeVar

T = TypeVar("T")


class Invertible(ABC, nn.Module):
    def forward(self, *args: T, **kwargs: T) -> Any:
        raise NotImplementedError

    def inverse(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, rev = False, **kwargs):
        if not rev:
            return self.forward(*args, **kwargs)

        return self.inverse(*args, **kwargs)
