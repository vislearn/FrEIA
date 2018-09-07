'''Subclasses of torch.nn.Module, that are reversible and can be used in the
nodes of the ReversibleGraphNet class. The only additional things that are
needed compared to the base class is an @staticmethod otuput_dims, and the
'rev'-argument of the forward-method.'''

from .fixed_transforms import *
from .reshapes import *
from .coupling_layers import *
from .graph_topology import *
from .coeff_functs import *
