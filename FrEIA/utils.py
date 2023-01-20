import torch

from typing import Callable, Any


def output_dims_compatible(invertible_module):
    """
    Hack to get output dimensions from any module as
    SequenceINN and GraphINN do not work with input/output shape API.
    """
    no_output_dims = (
            hasattr(invertible_module, "force_tuple_output")
            and not invertible_module.force_tuple_output
    )
    if not no_output_dims:
        return invertible_module.output_dims(invertible_module.dims_in)
    else:
        try:
            return invertible_module.output_dims(None)
        except TypeError:
            raise NotImplementedError(f"Can't determine output dimensions for {invertible_module.__class__}.")


def f_except(f: Callable, x: torch.Tensor, *dim, **kwargs):
    """ Apply f on all dimensions except those specified in dim """
    result = x
    dimensions = [d for d in range(x.dim()) if d not in dim]

    if not dimensions:
        raise ValueError(f"Cannot exclude dims {dim} from x with shape {x.shape}: No dimensions left.")

    return f(result, dim=dimensions, **kwargs)


def sum_except(x: torch.Tensor, *dim):
    """ Sum all dimensions of x except the ones specified in dim """
    return f_except(torch.sum, x, *dim)


def sum_except_batch(x):
    """ Sum all dimensions of x except the first (batch) dimension """
    return sum_except(x, 0)


def force_to(obj: Any, *args, **kwargs) -> Any:
    """
    Applies `.to()` on all tensors in the object structure of obj.
    Use this as long as https://github.com/pytorch/pytorch/issues/7795 is unresolved.
    """
    applied_cache = set()

    def _deep_to(obj):
        """ Applies `fn` to all tensors referenced in `obj` """
        if id(obj) in applied_cache:
            raise ValueError(f"Cannot call deep_to(...) on self-referential structure.")
        applied_cache.add(id(obj))

        if torch.is_tensor(obj):
            obj = obj.to(*args, **kwargs)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = _deep_to(value)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                obj[i] = _deep_to(value)
        elif isinstance(obj, tuple):
            obj = tuple(
                _deep_to(value)
                for value in obj
            )
        elif hasattr(obj, '__dict__'):
            obj = _deep_to(obj.__dict__)

        return obj

    return _deep_to(obj)
