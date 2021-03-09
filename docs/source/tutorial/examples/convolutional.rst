Convolutional INN with invertible downsampling
=================================================

For the following architecture (which works e.g. for CIFAR10), 3/4 of the
outputs are split off after some convolutions, which encode the local details,
and the rest are transformed further to encode semantic content.  This is
important, because even for moderately sized images, it becomes infeasible to
transform all dimenions through the full depth of the INN. Many dimensions will
just enocde image noise, so we can split them off early.
Because the computational graph contains multiple outputs, we have to use the full ``G`` machinery.

.. code:: python

  nodes = [Ff.InputNode(3, 32, 32, name='input')]
  ndim_x = 3 * 32 * 32

  # Higher resolution convolutional part
  for k in range(4):
      nodes.append(Ff.Node(nodes[-1],
                           Fm.GLOWCouplingBlock,
                           {'subnet_constructor':subnet_conv, 'clamp':1.2},
                           name=F'conv_high_res_{k}'))
      nodes.append(Ff.Node(nodes[-1],
                           Fm.PermuteRandom,
                           {'seed':k},
                           name=F'permute_high_res_{k}'))

  nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

  # Lower resolution convolutional part
  for k in range(12):
      if k%2 == 0:
          subnet = subnet_conv_1x1
      else:
          subnet = subnet_conv

      nodes.append(Ff.Node(nodes[-1],
                           Fm.GLOWCouplingBlock,
                           {'subnet_constructor':subnet, 'clamp':1.2},
                           name=F'conv_low_res_{k}'))
      nodes.append(Ff.Node(nodes[-1],
                           Fm.PermuteRandom,
                           {'seed':k},
                           name=F'permute_low_res_{k}'))

  # Make the outputs into a vector, then split off 1/4 of the outputs for the
  # fully connected part
  nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
  split_node = Ff.Node(nodes[-1],
                      Fm.Split,
                      {'section_sizes':(ndim_x // 4, 3 * ndim_x // 4), 'dim':0},
                      name='split')
  nodes.append(split_node)

  # Fully connected part
  for k in range(12):
      nodes.append(Ff.Node(nodes[-1],
                           Fm.GLOWCouplingBlock,
                           {'subnet_constructor':subnet_fc, 'clamp':2.0},
                           name=F'fully_connected_{k}'))
      nodes.append(Ff.Node(nodes[-1],
                           Fm.PermuteRandom,
                           {'seed':k},
                           name=F'permute_{k}'))

  # Concatenate the fully connected part and the skip connection to get a single output
  nodes.append(Ff.Node([nodes[-1].out0, split_node.out1],
                      Fm.Concat1d, {'dim':0}, name='concat'))
  nodes.append(Ff.OutputNode(nodes[-1], name='output'))

  conv_inn = Ff.GraphINN(nodes)


