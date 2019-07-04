import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm

ndim_total = 2 * 64 * 64

class CondNet(nn.Module):
    '''conditioning network'''
    def __init__(self):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        self.resolution_levels = nn.ModuleList([
                           nn.Sequential(nn.Conv2d(1,  64, 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(64,  128, 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.AvgPool2d(4),
                                         Flatten(),
                                         nn.Linear(2048, 512))])

    def forward(self, c):
        outputs = [c]
        for m in self.resolution_levels:
            outputs.append(m(outputs[-1]))
        return outputs[1:]

class ColorizationCINN(nn.Module):
    '''cINN, including the ocnditioning network'''
    def __init__(self, lr):
        super().__init__()

        self.cinn = self.build_inn()
        self.cond_net = CondNet()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.02 * torch.randn_like(p)

        self.trainable_parameters += list(self.cond_net.parameters())
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr)

    def build_inn(self):

        def sub_conv(ch_hidden, kernel):
            pad = kernel // 2
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                            nn.ReLU(),
                                            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

        def sub_fc(ch_hidden):
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Linear(ch_in, ch_hidden),
                                            nn.ReLU(),
                                            nn.Linear(ch_hidden, ch_out))

        nodes = [Ff.InputNode(2, 64, 64)]
        # outputs of the cond. net at different resolution levels
        conditions = [Ff.ConditionNode(64, 64, 64),
                      Ff.ConditionNode(128, 32, 32),
                      Ff.ConditionNode(128, 16, 16),
                      Ff.ConditionNode(512)]

        split_nodes = []

        subnet = sub_conv(32, 3)
        for k in range(2):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=conditions[0]))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        for k in range(4):
            subnet = sub_conv(64, 3 if k%2 else 1)

            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=conditions[1]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        #split off 6/8 ch
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections':[2,6], 'dim':0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        for k in range(4):
            subnet = sub_conv(128, 3 if k%2 else 1)

            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6},
                                 conditions=conditions[2]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        #split off 4/8 ch
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections':[4,4], 'dim':0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

        # fully_connected part
        subnet = sub_fc(512)
        for k in range(4):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6},
                                 conditions=conditions[3]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        # concat everything
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}))
        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)

    def forward(self, Lab):
        z = self.cinn(Lab[:,1:], c=self.cond_net(Lab[:,:1]))
        jac = self.cinn.log_jacobian(run_forward=False)
        return z, jac

    def reverse_sample(self, z, L):
        return self.cinn(z, c=self.cond_net(L), rev=True)
