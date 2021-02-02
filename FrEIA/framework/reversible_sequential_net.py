import torch.nn as nn
import torch

class ReversibleSequential(nn.Module):
    '''Simpler than FrEIA.framework.ReversibleGraphNet:
    Only supports a sequential series of modules (no splitting, merging, branching off).
    Has an append() method, to add new blocks in a more simple way than the computation-graph
    based approach of ReversibleGraphNet. For example:

    inn = ReversibleSequential(channels, dims_H, dims_W)

    for i in range(n_blocks):
        inn.append(FrEIA.modules.AllInOneBlock, clamp=2.0, permute_soft=True)
    inn.append(FrEIA.modules.HaarDownsampling)
    # and so on

    '''

    def __init__(self, *dims):
        super().__init__()

        self.shapes = [tuple(dims)]
        self.conditions = []
        self.module_list = nn.ModuleList()

    def append(self, module_class, cond=None, cond_shape=None, **kwargs):
        '''Append a reversible block from FrEIA.modules to the network.
        module_class: Class from FrEIA.modules.
        cond (int): index of which condition to use (conditions will be passed as list to forward()).
                    Conditioning nodes are not needed for ReversibleSequential.
        cond_shape (tuple[int]): the shape of the condition tensor.
        **kwargs: Further keyword arguments that are passed to the constructor of module_class (see example).
        '''

        dims_in = [self.shapes[-1]]
        self.conditions.append(cond)

        if cond is not None:
            kwargs['dims_c'] = [cond_shape]

        module = module_class(dims_in, **kwargs)
        self.module_list.append(module)
        ouput_dims = module.output_dims(dims_in)
        assert len(ouput_dims) == 1, "Module has more than one output"
        self.shapes.append(ouput_dims[0])


    def forward(self, x, c=None, rev=False):
        '''
        x (Tensor): input tensor (in contrast to ReversibleGraphNet, a list of tensors is not
                    supported, as ReversibleSequential only has one input).
        c (list[Tensor]): list of conditions.
        rev: whether to compute the network forward or reversed.

        Returns
        z (Tensor): network output.
        jac (Tensor): log-jacobian-determinant.
        There is no separate log_jacobian() method, it is automatically computed during forward().
        '''

        iterator = range(len(self.module_list))
        jac = 0

        if rev:
            iterator = reversed(iterator)

        for i in iterator:
            if self.conditions[i] is None:
                x, j = (self.module_list[i]([x], rev=rev)[0],
                        self.module_list[i].jacobian(x, rev=rev))
            else:
                x, j = (self.module_list[i]([x], c=[c[self.conditions[i]]], rev=rev)[0],
                        self.module_list[i].jacobian(x, c=[c[self.conditions[i]]], rev=rev))
            jac = j + jac

        return x, jac
