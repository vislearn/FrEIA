Quickstart guide
=====================================

To jump straight into the code, see this basic usage example, which learns and
then samples from the moons dataset provided in ``sklearn``. For information on
the structure of FrEIA, as well as more detailed examples of networks,
including custom invertible operations, see the full tutorial below.

.. code:: python

  # standard imports
  import torch
  import torch.nn as nn
  from sklearn.datasets import make_moons

  # FrEIA imports
  import FrEIA.framework as Ff
  import FrEIA.modules as Fm

  BATCHSIZE = 100
  N_DIM = 2

  # we define a subnet for use inside an affine coupling block
  # for more detailed information see the full tutorial
  def subnet_fc(dims_in, dims_out):
      return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(),
                           nn.Linear(512,  dims_out))

  # a simple chain of operations is collected by ReversibleSequential
  inn = Ff.SequenceINN(N_DIM)
  for k in range(8):
      inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

  optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)

  # a very basic training loop
  for i in range(1000):
      optimizer.zero_grad()
      # sample data from the moons distribution
      data, label = make_moons(n_samples=BATCHSIZE, noise=0.05)
      x = torch.Tensor(data)
      # pass to INN and get transformed variable z and log Jacobian determinant
      z, log_jac_det = inn(x)
      # calculate the negative log-likelihood of the model with a standard normal prior
      loss = 0.5*torch.sum(z**2, 1) - log_jac_det
      loss = loss.mean() / N_DIM
      # backpropagate and update the weights
      loss.backward()
      optimizer.step()

  # sample from the INN by sampling from a standard normal and transforming
  # it in the reverse direction
  z = torch.randn(BATCHSIZE, N_DIM)
  samples, _ = inn(z, rev=True)



