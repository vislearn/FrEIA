import os

import torch

from typing import Callable, Any, Tuple, Union, Iterable

from torch.utils.data import TensorDataset, DataLoader


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
    applied_stack = set()

    def _deep_to(obj):
        """ Applies `fn` to all tensors referenced in `obj` """
        obj_id = id(obj)
        if obj_id in applied_stack:
            raise ValueError(f"Cannot call deep_to(...) on self-referential structure.")
        applied_stack.add(obj_id)

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
            _deep_to(obj.__dict__)

        applied_stack.remove(obj_id)
        return obj

    return _deep_to(obj)


def tuple_free_forward(module, data: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    try:
        is_tuple_module = module.force_tuple_output
    except AttributeError:
        is_tuple_module = True

    if is_tuple_module:
        data = data,
    out, jac = module(data, *args, **kwargs)

    if is_tuple_module:
        out, = out
    return out, jac


def tuple_free_batch_forward(module, data: torch.Tensor, batch_size, loader_kwargs=None, device=None, **forward_kwargs) -> \
        Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """
    Executes a module on the passed data in batches.

    A dataloader is used to push the data to cuda if the data is on the cpu.
    You can specify workers etc. via loader_kwargs.
    """
    if loader_kwargs is None:
        loader_kwargs = dict()

    target_device = data.device
    outs = []
    jacs = []

    if data.device == torch.device("cpu"):
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, **loader_kwargs)
    else:
        if len(loader_kwargs) == 0:
            raise ValueError("Can't use loader_kwargs with non-cpu data.")
        dataloader = data.split(batch_size)

    for (batch,) in dataloader:
        out, jac = tuple_free_forward(module, batch.to(device), **forward_kwargs)
        outs.append(out.to(target_device))
        if jac is None:
            jacs.append(None)
        else:
            jacs.append(jac.to(target_device))

    out_cat = torch.cat(outs)
    if all(jac is not None for jac in jacs):
        jac_cat = torch.cat(jacs)
    else:
        jac_cat = None
    return out_cat, jac_cat


def _get_node_type_for_plotting(n):
    node_type = n.module
    if node_type == None:
        node_type = n.__repr__().split(" ")[0]
    else:
        node_type = node_type._get_name()
    return node_type


def _reverse_edges(edges):
    rev_edges = {}
    for node_out, node_ins in edges.items():
        for node_in in node_ins:
            rev_edges[node_in] = node_out

    return rev_edges


def _get_edges(nodes, rev=False):
    edges_out_to_in = {node_b: [node_a for node_a in node_b.inputs] for
                       node_b in nodes if node_b.inputs}

    cond_edges_out_to_in = {node_b: [node_a for node_a in node_b.conditions] for
                            node_b in nodes if node_b.conditions}

    if not rev:
        edges = _reverse_edges(edges_out_to_in)
        cond_edges = _reverse_edges(cond_edges_out_to_in)
    else:
        edges = edges_out_to_in
        cond_edges = cond_edges_out_to_in

    return edges, cond_edges


def plot_graph(nodes: Iterable, path: str, filename: str) -> None:
    """
    Generates a plot of the GraphINN and stores it as pdf and dot file

    Parameters:
        path: Directory to store the plots in. Must exist previous to plotting
        filename: Name of the newly generated plots
    """
    if not os.path.exists(path):
        raise Exception("Path %s does not exist." % path)

    import graphviz as g

    G = g.Digraph()
    for n in nodes:
        node_type = _get_node_type_for_plotting(n)
        G.node(str(id(n)), node_type)

    edges, cond_edges = _get_edges(nodes, rev=True)

    for key, value in edges.items():
        for idx, v in enumerate(value):
            dims = key.input_dims[idx]
            label = '(' + ','.join(str(d) for d in dims) + ')'
            G.edge(str(id(v[0])), str(id(key)), label=label)

    for key, value in cond_edges.items():
        for idx, v in enumerate(value):
            dims = v.output_dims[0]
            label = '(' + ','.join(str(d) for d in dims) + ')'
            G.edge(str(id(v)), str(id(key)), label=label)

    file_path = os.path.abspath(os.path.join(path, filename))
    try:
        G.render(file_path)
    except g.backend.execute.ExecutableNotFound:
        raise Exception(
            "Skipped plotting graph since graphviz backend is not installed. "
            "Try installing it via 'sudo apt-get install graphviz'"
        )
