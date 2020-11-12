'''Subclasses of torch.nn.Module, that are reversible and can be used in the
nodes of the ReversibleGraphNet class. The only additional things that are
needed compared to the base class is an @staticmethod otuput_dims, and the
'rev'-argument of the forward-method.

Coupling blocks:

* NICECouplingBlock
* RNVPCouplingBlock
* GLOWCouplingBlock
* GINCouplingBlock
* AffineCouplingOneSided
* ConditionalAffineTransform

Other learned transforms:

* ActNorm
* IResNetLayer
* InvAutoAct
* InvAutoActFixed
* InvAutoActTwoSided
* InvAutoConv2D
* InvAutoFC
* LearnedElementwiseScaling
* OrthogonalTransform
* HouseholderPerm

Fixed (non-learned) transforms:

* PermuteRandom
* FixedLinearTransform
* Fixed1x1Conv

Graph topology:


* SplitChannel
* ConcatChannel
* Split1D
* Concat1d

Reshaping:

* IRevNetDownsampling
* IRevNetUpsampling
* HaarDownsampling
* HaarUpsampling',
* Flatten
* Reshape

'''

from .fixed_transforms import *
from .reshapes import *
from .coupling_layers import *
from .graph_topology import *
from .coeff_functs import *
from .orthogonal import *
from .inv_auto_layers import *
from .invertible_resnet import *
from .gaussian_mixture import *

__all__ = [
            'glow_coupling_layer',
            'rev_layer',
            'rev_multiplicative_layer',
            'AffineCoupling',
            'ExternalAffineCoupling',
            'ActNorm',
            'HouseholderPerm',
            'IResNetLayer',
            'InvAutoAct',
            'InvAutoActFixed',
            'InvAutoActTwoSided',
            'InvAutoConv2D',
            'InvAutoFC',
            'LearnedElementwiseScaling',
            'orthogonal_layer',
            'conv_1x1',
            'linear_transform',
            'permute_layer',
            'split_layer',
            'cat_layer',
            'channel_split_layer',
            'channel_merge_layer',
            'reshape_layer',
            'flattening_layer',
            'haar_multiplex_layer',
            'haar_restore_layer',
            'i_revnet_downsampling',
            'i_revnet_upsampling',
            'F_conv',
            'F_fully_connected',
            'F_fully_convolutional',
            'NICECouplingBlock',
            'RNVPCouplingBlock',
            'GLOWCouplingBlock',
            'GINCouplingBlock',
            'AffineCouplingOneSided',
            'ConditionalAffineTransform',
            'PermuteRandom',
            'FixedLinearTransform',
            'Fixed1x1Conv',
            'SplitChannel',
            'ConcatChannel',
            'Split1D',
            'Concat1d',
            'OrthogonalTransform',
            'HouseholderPerm',
            'IRevNetDownsampling',
            'IRevNetUpsampling',
            'HaarDownsampling',
            'HaarUpsampling',
            'Flatten',
            'Reshape',
            'GaussianMixtureModel',
            ]
