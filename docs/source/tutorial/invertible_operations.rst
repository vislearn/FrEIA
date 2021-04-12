Invertible Operations
=======================

A number of commonly used invertible operations are provided in the ``FrEIA.modules`` submodule.
They are documented in detail `here <https://vll-hd.github.io/FrEIA/_build/html/FrEIA.modules.html>`.

Coupling blocks
**************************

All coupling blocks (GLOW, RNVP, NICE), merit special discussion, because
they are the most used invertible transforms.

* The coupling blocks contain smaller feed-forward subnetworks predicting the affine coefficients.
  The in- and output shapes of the subnetworks depend on the in- output size of the coupling block itself.
  These size are not known when coding the INN (or perhaps can be worked out by
  hand, but would have to be worked out anew every time the architecture is modified slightly).
  Therefore, the subnetworks can not be directly passed as ``nn.Modules``, but
  rather in the form of a function or class, that constructs the subnetworks
  given in- and output size. This is a lot simpler than it sounds, for a fully connected subnetwork we could use for example:

  .. code:: python

    def fc_constr(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, 128), nn.ReLU(),
                            nn.Linear(128,  128), nn.ReLU(),
                            nn.Linear(128,  dims_out))

* The RNVP and GLOW coupling blocks have an additional hyperparameter ``clamp``.
  This is becuase, instead of the exponential function ``exp(s)``, we use ``exp( 2*c/pi * atan(x))``
  in the coupling blocks (``clamp``-parameter ``c``).
  This leads to much more stable training and enables larger learning rates.
  Effectively, the multiplication component of the coupling block is limited between ``exp(c)`` and ``1/exp(c)``.
  The Jacobian determinant is thereby limited between ``±D*c`` (dimensionality of data ``D``).
  In general, ``clamp = 2.0`` is a good place to start:

  .. code:: python

    glow = Node([in1.out0], GLOWCouplingBlock,
                {'subnet_constructor': fc_constr, 'clamp': 2.0},
                name='GLOW coupling block')

* The module ``FrEIA.modules.AllInOneBlock`` not only includes glow coupling, but also a learned global scaling ('ActNorm'),
  and a random permutation, along with various other features.
  This saves implementation effort, as these three operations are usually used together.
  See `here <https://vll-hd.github.io/FrEIA/_build/html/FrEIA.modules.html#coupling-blocks>` for details.


Defining custom invertible operations
**************************************

Custom invertible modules can be written as extensions of the
``Fm.InvertibleModule`` base class. Refer to the documentation of this class
for detailed information on requirements. 

Below are two simple examples which illustrate the definition and use of custom
modules and can be used as basic templates.  The first multiplies each
dimension of an input tensor by either 1 or 2, chosen in a random but fixed
way.  The second is a conditional operation which takes two inputs and swaps
them if the condition is positive, doing nothing otherwise.

Notes:

* The ``Fm.InvertibleModule`` must be initialized with the ``dims_in`` argument
  and optionally ``dims_c`` if there is a conditioning input.  
* ``forward`` should return a tuple of outputs (even if there is only one), with additional
  ``log_jac_det`` term. This Jacobian term can be, but does not need to be
  calculated if ``jac=False``.

Definition:

.. code:: python

  class FixedRandomElementwiseMultiply(Fm.InvertibleModule):

      def __init__(self, dims_in):
          super().__init__(dims_in)
          self.random_factor = torch.randint(1, 3, size=(1, dims_in[0][0]))
          
      def forward(self, x, rev=False, jac=True):
          # the Jacobian term is trivial to calculate so we return it
          # even if jac=False
          
          # x is passed to the function as a list (in this case of only on element)
          x = x[0]
          if not rev:
              # forward operation
              x = x * self.random_factor
              log_jac_det = self.random_factor.float().log().sum()
          else:
              # backward operation
              x = x / self.random_factor
              log_jac_det = -self.random_factor.float().log().sum()
          
          return (x,), log_jac_det
      
      def output_dims(self, input_dims):
          return input_dims

          
          
          
  class ConditionalSwap(Fm.InvertibleModule):

      def __init__(self, dims_in, dims_c):
          super().__init__(dims_in, dims_c=dims_c)
          
      def forward(self, x, c, rev=False, jac=True):
          # in this case, the forward and reverse operations are identical
          # so we don't use the rev argument
          x1, x2 = x
          log_jac_det = 0.
          
          # make copies of the inputs
          x1_new = x1 + 0.
          x2_new = x2 + 0.
          
          for i in range(x1.size(0)):
              x1_new[i] = x1[i] if c[0][i] > 0 else x2[i]
              x2_new[i] = x2[i] if c[0][i] > 0 else x1[i]

          return (x1_new, x2_new), log_jac_det
      
      def output_dims(self, input_dims):
          return input_dims


Basic Usage Example:

.. code:: python

  BATCHSIZE = 10
  DIMS_IN = 2

  # build up basic net using SequenceINN
  net = Ff.SequenceINN(DIMS_IN)
  for i in range(2):
      net.append(FixedRandomElementwiseMultiply)

  # define inputs
  x = torch.randn(BATCHSIZE, DIMS_IN)

  # run forward
  z, log_jac_det = net(x)

  # run in reverse
  x_rev, log_jac_det_rev = net(z, rev=True)



More Complicated Example:

.. code:: python

  BATCHSIZE = 10
  DIMS_IN = 2

  # define a graph INN

  input_1 = Ff.InputNode(DIMS_IN, name='input_1')
  input_2 = Ff.InputNode(DIMS_IN, name='input_2')

  cond = Ff.ConditionNode(1, name='condition')

  mult_1 = Ff.Node(input_1.out0, FixedRandomElementwiseMultiply, {}, name='mult_1')
  cond_swap = Ff.Node([mult_1.out0, input_2.out0], ConditionalSwap, {}, conditions=cond, name='conditional_swap')
  mult_2 = Ff.Node(cond_swap.out1, FixedRandomElementwiseMultiply, {}, name='mult_2')

  output_1 = Ff.OutputNode(cond_swap.out0, name='output_1')
  output_2 = Ff.OutputNode(mult_2.out0, name='output_2')

  net = Ff.GraphINN([input_1, input_2, cond, mult_1, cond_swap, mult_2, output_1, output_2])

  # define inputs
  x1 = torch.randn(BATCHSIZE, DIMS_IN)
  x2 = torch.randn(BATCHSIZE, DIMS_IN)
  c = torch.randn(BATCHSIZE)

  # run forward
  (z1, z2), log_jac_det = net([x1, x2], c=c)

  # run in reverse without necessarily calculating Jacobian term (i.e. jac=False)
  (x1_rev, x2_rev), _ = net([z1, z2], c=c, rev=True, jac=False)

