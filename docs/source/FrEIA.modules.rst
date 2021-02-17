FrEIA.modules package
=====================

.. automodule:: FrEIA.modules


Abstract template
-----------------

.. autoclass:: InvertibleModule


Coupling blocks
---------------

.. autoclass:: AllInOneBlock

.. autoclass:: NICECouplingBlock

.. autoclass:: RNVPCouplingBlock

.. autoclass:: GLOWCouplingBlock

.. autoclass:: GINCouplingBlock

.. autoclass:: AffineCouplingOneSided

.. autoclass:: ConditionalAffineTransform


Other learned transforms
------------------------

.. autoclass:: ActNorm

.. autoclass:: IResNetLayer

.. autoclass:: InvAutoAct

.. autoclass:: InvAutoActFixed

.. autoclass:: InvAutoActTwoSided

.. autoclass:: InvAutoConv2D

.. autoclass:: InvAutoFC

.. autoclass:: LearnedElementwiseScaling

.. autoclass:: OrthogonalTransform

.. autoclass:: HouseholderPerm

.. autoclass:: GaussianMixtureModel


Fixed (non-learned) transforms
------------------------------

.. autoclass:: PermuteRandom

.. autoclass:: FixedLinearTransform

.. autoclass:: Fixed1x1Conv


Graph topology
--------------

.. autoclass:: SplitChannel

.. autoclass:: ConcatChannel

.. autoclass:: Split

.. autoclass:: Concat


Reshaping
---------

.. autoclass:: IRevNetDownsampling

.. autoclass:: IRevNetUpsampling

.. autoclass:: HaarDownsampling

.. autoclass:: HaarUpsampling

.. autoclass:: Flatten

.. autoclass:: Reshape
