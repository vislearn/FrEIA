FrEIA.modules package
=====================

.. automodule:: FrEIA.modules
    :no-members:


Abstract template
-----------------

.. autoclass:: InvertibleModule
    :exclude-members: training
    :members: forward, output_dims



Coupling blocks
---------------

.. autoclass:: AllInOneBlock

.. autoclass:: NICECouplingBlock

.. autoclass:: RNVPCouplingBlock

.. autoclass:: GLOWCouplingBlock

.. autoclass:: GINCouplingBlock

.. autoclass:: AffineCouplingOneSided

.. autoclass:: ConditionalAffineTransform


Reshaping
---------

.. autoclass:: IRevNetDownsampling

.. autoclass:: IRevNetUpsampling

.. autoclass:: HaarDownsampling

.. autoclass:: HaarUpsampling

.. autoclass:: Flatten

.. autoclass:: Reshape


Graph topology
--------------

.. autoclass:: Split

.. autoclass:: Concat


Other learned transforms
------------------------

.. autoclass:: ActNorm

.. autoclass:: IResNetLayer

.. autoclass:: InvAutoAct

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
