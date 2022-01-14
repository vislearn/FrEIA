Sequential API
=====================================

In many cases, the INN will just consist of a sequence of invertible operations with one input and one output 
without any splitting or merging in between.
For this simple case, we provide ``FrEIA.framework.SequenceINN``. Here, invertible operations can be added to the INN
through the ``.append()``-method, without having to explicitly write out a computation graph.

.. code:: python

    import torch.nn as nn
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm

    def subnet_fc(dims_in, dims_out):
        '''Return a feed-forward subnetwork, to be used in the coupling blocks below'''
        return nn.Sequential(nn.Linear(dims_in, 128), nn.ReLU(),
                             nn.Linear(128,  128), nn.ReLU(),
                             nn.Linear(128,  dims_out))

    # a tuple of the input data dimension. 784 is the dimension of flattened MNIST images.
    # (input_dims would be (3, 32, 32) for CIFAR for instance)
    input_dims = (784,)

    # construct the INN (not containing any operations so far)
    inn = Ff.SequenceINN(*input_dims)

    # append coupling blocks to the sequence of operations
    for k in range(8):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc)

The INN contain 8 ``AllInOneBlock`` coupling blocks in a sequence.
Given an input ``x``, the output ``z`` together with  the log-determinant-Jacobian ``jac``
is simply obtained by calling the INN: ``z, jac = inn(x)``.
The inverse can be computed using the ``rev`` keyword:

.. code:: python

    z, jac = inn(x)
    x_inv, jac_inv = inn(z, rev=True)

    # inverting from the outputs should give the original inputs again
    assert torch.max(torch.abs(x - x_inv)) < 1e-5

    # the inverse log-det-Jacobian should be the negative of the forward log-det-Jacobian
    assert torch.max(torch.abs(jac + jac_inv)) < 1e-5

The ``SequenceINN`` is a child class of ``torch.nn.Module``, so all pytorch methods will work
(``.cuda()``, ``.state_dict()``, etc.).

For **conditional** sequential architectures, we presuppose a known list of
conditions, and that each invertible operation only receives one condition.
If this is not the case, we refer to ``FrEIA.framework.GraphINN`` (see next page).
For now, we imagine that the only condition is a one-hot label of the MNIST class.
A conditional INN would then be constructed as follows:

.. code:: python

    cond_dims = (10,)

    # use the input_dims (784,) from above
    cinn = Ff.SequenceINN(*input_dims)

    for k in range(8):
        # The cond=0 argument tells the operation which of the conditions it should
        # use, that are supplied with the call. So cond=0 means 'use the first condition'
        # (there is only one condition in this case).
        cinn.append(Fm.AllInOneBlock, cond=0, cond_shape=cond_dims, subnet_constructor=subnet_fc)

    # the conditions have to be given as a list (in this example, a list with
    # one entry, 'one_hot_labels').  In general, multiple conditions can be
    # given. The cond argument of the append() method above specifies which
    # condition is used for each operation.
    z, jac = cinn(x, c=[one_hot_labels])


