Small fully-connected INNs
=====================================

.. code:: python

  # These imports and declarations apply to all examples
  import torch.nn as nn

  import FrEIA.framework as Ff
  import FrEIA.modules as Fm

  def subnet_fc(c_in, c_out):
      return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                          nn.Linear(512,  c_out))

  def subnet_conv(c_in, c_out):
      return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                          nn.Conv2d(256,  c_out, 3, padding=1))

  def subnet_conv_1x1(c_in, c_out):
      return nn.Sequential(nn.Conv2d(c_in, 256,   1), nn.ReLU(),
                          nn.Conv2d(256,  c_out, 1))

Simple INN in 2 dimensions
****************************

The following INN only has 2 input dimensions.
It should be able to learn to generate most 2D distributions (gaussian mixtures, different shapes, ...),
and can be easily visualized.
We will use a series of ``AllInOneBlock`` operations, which combine affine coupling, a permutation and ActNorm in a single structure.
Since the computation graph is a simple chain of operations, we can define the network using the ``SequenceINN`` API.

.. code:: python

  inn = Ff.SequenceINN(2)
  for k in range(8):
      inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

Conditional INN for MNIST
***************************

The following cINN is able to perform conditional MNIST generation quite well.
Note that is is not particularly efficient, with respect to the number of parameters (see convolutional INN for that).
Again, we use a chain of ``AllInOneBlock``s, collected together by ``SequenceINN``.

.. code:: python

  cinn = Ff.SequenceINN(28*28)
  for k in range(12):
      cinn.append(Fm.AllInOneBlock, cond=0, cond_shape=(10,), subnet_constructor=subnet_fc)

