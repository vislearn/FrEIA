import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):


    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.bw_conv1_1 = self.__conv(2, name='bw_conv1_1', in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv1_2 = self.__conv(2, name='conv1_2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=True, dilation=1, padding=1)
        self.conv1_2norm = self.__batch_normalization(2, 'conv1_2norm', num_features=64, eps=9.999999747378752e-06, momentum=0.1)
        self.conv2_1 = self.__conv(2, name='conv2_1', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv2_2 = self.__conv(2, name='conv2_2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=True, dilation=1, padding=1)
        self.conv2_2norm = self.__batch_normalization(2, 'conv2_2norm', num_features=128, eps=9.999999747378752e-06, momentum=0.1)
        self.conv3_1 = self.__conv(2, name='conv3_1', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv3_2 = self.__conv(2, name='conv3_2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv3_3 = self.__conv(2, name='conv3_3', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=True, dilation=1, padding=1)
        self.conv3_3norm = self.__batch_normalization(2, 'conv3_3norm', num_features=256, eps=9.999999747378752e-06, momentum=0.1)
        self.conv4_1 = self.__conv(2, name='conv4_1', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv4_2 = self.__conv(2, name='conv4_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv4_3 = self.__conv(2, name='conv4_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv4_3norm = self.__batch_normalization(2, 'conv4_3norm', num_features=512, eps=9.999999747378752e-06, momentum=0.1)
        self.conv5_1 = self.__conv(2, name='conv5_1', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=2, padding=2)
        self.conv5_2 = self.__conv(2, name='conv5_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=2, padding=2)
        self.conv5_3 = self.__conv(2, name='conv5_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=2, padding=2)
        self.conv5_3norm = self.__batch_normalization(2, 'conv5_3norm', num_features=512, eps=9.999999747378752e-06, momentum=0.1)
        self.conv6_1 = self.__conv(2, name='conv6_1', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=2, padding=2)
        self.conv6_2 = self.__conv(2, name='conv6_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=2, padding=2)
        self.conv6_3 = self.__conv(2, name='conv6_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=2, padding=2)
        self.conv6_3norm = self.__batch_normalization(2, 'conv6_3norm', num_features=512, eps=9.999999747378752e-06, momentum=0.1)
        self.conv7_1 = self.__conv(2, name='conv7_1', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv7_2 = self.__conv(2, name='conv7_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv7_3 = self.__conv(2, name='conv7_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv7_3norm = self.__batch_normalization(2, 'conv7_3norm', num_features=512, eps=9.999999747378752e-06, momentum=0.1)
        self.conv8_1 = self.__conv_transpose(2, name='conv8_1', in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(2, 2), groups=1, bias=True)
        self.conv8_2 = self.__conv(2, name='conv8_2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv8_3 = self.__conv(2, name='conv8_3', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True, dilation=1, padding=1)
        self.conv8_313 = self.__conv(2, name='conv8_313', in_channels=256, out_channels=313, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True, dilation=1, padding=0)
        self.class8_ab = self.__conv(2, name='class8_ab', in_channels=313, out_channels=2, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True, dilation=1, padding=0)

    def features(self, x):
        out = self.bw_conv1_1(x)
        out = F.relu(out)
        out = self.conv1_2(out)
        out = F.relu(out)
        out = self.conv1_2norm(out)
        out = self.conv2_1(out)
        out = F.relu(out)
        out = self.conv2_2(out)
        out = F.relu(out)
        out = self.conv2_2norm(out)
        out = self.conv3_1(out)
        out = F.relu(out)
        out = self.conv3_2(out)
        out = F.relu(out)
        out = self.conv3_3(out)
        out = F.relu(out)
        out = self.conv3_3norm(out)
        out = self.conv4_1(out)
        out = F.relu(out)
        out = self.conv4_2(out)
        out = F.relu(out)
        out = self.conv4_3(out)
        out = F.relu(out)
        out = self.conv4_3norm(out)
        out = self.conv5_1(out)
        out = F.relu(out)
        out = self.conv5_2(out)
        out = F.relu(out)
        out = self.conv5_3(out)
        out = F.relu(out)
        out = self.conv5_3norm(out)
        out = self.conv6_1(out)
        out = F.relu(out)
        out = self.conv6_2(out)
        out = F.relu(out)
        out = self.conv6_3(out)
        out = F.relu(out)
        out = self.conv6_3norm(out)
        out = self.conv7_1(out)
        out = F.relu(out)
        out = self.conv7_2(out)
        out = F.relu(out)
        out = self.conv7_3(out)
        out = F.relu(out)
        out = self.conv7_3norm(out)
        out = self.conv8_1(out)
        out = F.relu(out)
        out = self.conv8_2(out)
        out = F.relu(out)
        out = self.conv8_3(out)

        return out

    def forward(self, x):
        out = self.features(x)
        out = F.relu(out)
        out = self.conv8_313(out)
        out = 2.606 * out
        out = F.softmax(out, dim=1)
        out = self.class8_ab(out)

        return out

    def fwd_from_features(self, f):
        out = F.relu(f)
        out = self.conv8_313(out)
        out = 2.606 * out
        out = F.softmax(out, dim=1)
        out = self.class8_ab(out)

        return out

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        try:
            if 'scale' in __weights_dict[name]:
                layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
            else:
                layer.weight.data.fill_(1)

            if 'bias' in __weights_dict[name]:
                layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
            else:
                layer.bias.data.fill_(0)

            layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
            layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        except:
            pass
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        try:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
            if 'bias' in __weights_dict[name]:
                layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        except:
            pass
        return layer

    @staticmethod
    def __conv_transpose(dim, name, **kwargs):
        if   dim == 1:  layer = nn.ConvTranspose1d(**kwargs)
        elif dim == 2:  layer = nn.ConvTranspose2d(**kwargs)
        elif dim == 3:  layer = nn.ConvTranspose3d(**kwargs)
        else:           raise NotImplementedError()

        try:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
            if 'bias' in __weights_dict[name]:
                layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        except:
            pass
        return layer
