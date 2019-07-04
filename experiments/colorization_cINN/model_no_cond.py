import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from FrEIA.framework import *
from FrEIA.modules import *
from cbn_layer import *
from subnet_coupling import *
import data
import config as c

n_blocks_fc = 8
outputs = []

conditions = [ConditionNode(1, c.img_dims[0], c.img_dims[1])]

def random_orthog(n):
    w = np.random.randn(n, n)
    w = w + w.T
    w, S, V = np.linalg.svd(w)
    return torch.FloatTensor(w)

class HaarConv(nn.Module):

    def __init__(self, level):
        super().__init__()

        self.in_channels = 4**level
        self.fac_fwd = 0.25
        self.haar_weights = torch.ones(4,1,2,2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x):
        out = F.conv2d(x, self.haar_weights,
                       bias=None, stride=2, groups=self.in_channels)
        return out * self.fac_fwd

def cond_subnet(level):
    return nn.Sequential(*[HaarConv(i) for i in range(level+2)])

def _add_conditioned_section(nodes, depth, channels_in, channels, cond_level):

    for k in range(depth):
        nodes.append(Node([nodes[-1].out0],
                          subnet_coupling_layer,
                          {'clamp':c.clamping, 'F_class':F_conv,
                           'subnet':cond_subnet(cond_level), 'sub_len':4**(cond_level+2),
                           'F_args':{'leaky_slope': 5e-2, 'channels_hidden':channels}},
                          conditions=[conditions[0]], name=F'conv_{k}'))

        nodes.append(Node([nodes[-1].out0], conv_1x1, {'M':random_orthog(channels_in)}))


def _add_split_downsample(nodes, split, downsample, channels_in, channels):
    if downsample=='haar':
        nodes.append(Node([nodes[-1].out0], haar_multiplex_layer, {'rebalance':0.5, 'order_by_wavelet':True}, name='haar'))
    if downsample=='reshape':
        nodes.append(Node([nodes[-1].out0], i_revnet_downsampling, {}, name='reshape'))

    for i in range(2):
        nodes.append(Node([nodes[-1].out0], conv_1x1, {'M':random_orthog(channels_in*4)}))
        nodes.append(Node([nodes[-1].out0],
                      glow_coupling_layer,
                      {'clamp':c.clamping, 'F_class':F_conv,
                       'F_args':{'kernel_size':1, 'leaky_slope': 1e-2, 'channels_hidden':channels}},
                      conditions=[]))

    if split:
        nodes.append(Node([nodes[-1].out0], split_layer,
                        {'split_size_or_sections': split, 'dim':0}, name='split'))

        output = Node([nodes[-1].out1], flattening_layer, {}, name='flatten')
        nodes.insert(-2, output)
        nodes.insert(-2, OutputNode([output.out0], name='out'))

def _add_fc_section(nodes):
    nodes.append(Node([nodes[-1].out0], flattening_layer, {}, name='flatten'))
    for k in range(n_blocks_fc):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed':k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                {'clamp':c.clamping, 'F_class':F_fully_connected, 'F_args':{'internal_size':512}},
                conditions=[], name=F'fc_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='out'))

nodes = [InputNode(2, *c.img_dims, name='inp')]
# 2x64x64 px
_add_conditioned_section(nodes, depth=4, channels_in=2, channels=32, cond_level=0)
_add_split_downsample(nodes, split=False, downsample='reshape', channels_in=2, channels=64)

# 8x32x32 px
_add_conditioned_section(nodes, depth=6, channels_in=8, channels=64, cond_level=1)
_add_split_downsample(nodes, split=(16, 16), downsample='reshape', channels_in=8, channels=128)

# 16x16x16 px
_add_conditioned_section(nodes, depth=6, channels_in=16, channels=128, cond_level=2)
_add_split_downsample(nodes, split=(32, 32), downsample='reshape', channels_in=16, channels=256)

# 32x8x8 px
_add_conditioned_section(nodes, depth=6, channels_in=32, channels=256, cond_level=3)
_add_split_downsample(nodes, split=(32, 3*32), downsample='haar', channels_in=32, channels=256)

# 32x4x4 = 512 px
_add_fc_section(nodes)

def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if len(split) > 3 and split[3][-1] == '3': # last convolution in the coeff func
                param.data.fill_(0.)


cinn = ReversibleGraphNet(nodes + conditions, verbose=False)
output_dimensions = []
for o in nodes:
    if type(o) is OutputNode:
        output_dimensions.append(o.input_dims[0][0])

cinn.cuda()
init_model(cinn)

if c.load_inn_only:
    cinn.load_state_dict(torch.load(c.load_inn_only)['net'])

class DummyFeatureNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dumm_param = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        return x
    def features(self, x):
        return x

efros_net = DummyFeatureNet()

def prepare_batch(x):

    net_feat = combined_model.module.feature_network
    net_inn  = combined_model.module.inn
    net_cond = combined_model.module.fc_cond_network

    with torch.no_grad():
        x = x.cuda()
        x_l, x_ab = x[:, 0:1], x[:, 1:]

        x_ab = F.interpolate(x_ab, size=c.img_dims)
        x_ab += 5e-2 * torch.cuda.FloatTensor(x_ab.shape).normal_()

        cond = [x_l]
        ab_pred = None

    return x_l.detach(), x_ab.detach(), cond, ab_pred

class WrappedModel(nn.Module):
    def __init__(self, feature_network, fc_cond_network, inn):
        super().__init__()

        self.feature_network = feature_network
        self.fc_cond_network = fc_cond_network
        self.inn = inn

    def forward(self, x):

        x_l, x_ab = x[:, 0:1], x[:, 1:]

        x_ab = F.interpolate(x_ab, size=c.img_dims)
        x_ab += 5e-2 * torch.cuda.FloatTensor(x_ab.shape).normal_()

        cond = [x_l]

        z = self.inn(x_ab, cond)
        zz = sum(torch.sum(o**2, dim=1) for o in z)
        jac = self.inn.jacobian(run_forward=False)

        return zz, jac

    def reverse_sample(self, z, cond):
        return self.inn(z, cond, rev=True)

combined_model = WrappedModel(efros_net, None, cinn)
combined_model.cuda()
combined_model = nn.DataParallel(combined_model, device_ids=c.device_ids)

params_trainable = list(filter(lambda p: p.requires_grad, combined_model.module.inn.parameters()))

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

sched_factor = 0.2
sched_patience = 8
sched_trehsh = 0.001
sched_cooldown = 2

weight_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                            factor=sched_factor,
                                                            patience=sched_patience,
                                                            threshold=sched_trehsh,
                                                            min_lr=0, eps=1e-08,
                                                            cooldown=sched_cooldown,
                                                            verbose = True)

weight_scheduler_fixed = torch.optim.lr_scheduler.torch.optim.lr_scheduler.StepLR(optim, 120, gamma=0.2)

class DummyOptim:
    def __init__(self):
        self.param_groups = []
    def state_dict(self):
        return {}
    def load_state_dict(self, *args, **kwargs):
        pass
    def step(self, *args, **kwargs):
        pass
    def zero_grad(self):
        pass

efros_net.train()

if c.end_to_end:
    feature_optim = torch.optim.Adam(combined_model.module.feature_network.parameters(), lr=c.lr_feature_net, betas=c.betas, eps=1e-4)
    feature_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(feature_optim,
                                                            factor=sched_factor,
                                                            patience=sched_patience,
                                                            threshold=sched_trehsh,
                                                            min_lr=0, eps=1e-08,
                                                            cooldown=sched_cooldown,
                                                            verbose = True)
else:
    feature_optim = DummyOptim()
    feature_scheduler = DummyOptim()

def optim_step():
    optim.step()
    optim.zero_grad()

    feature_optim.step()
    feature_optim.zero_grad()

def save(name):
    torch.save({'opt':optim.state_dict(),
                'opt_f':feature_optim.state_dict(),
                'net':combined_model.state_dict()}, name)

def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    combined_model.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
        feature_optim.load_state_dict(state_dicts['opt_f'])
    except:
        print('Cannot load optimizer for some reason or other')
