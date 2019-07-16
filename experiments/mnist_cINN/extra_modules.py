from math import exp
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from FrEIA.modules import *

class F_fully_conv(nn.Module):

    def __init__(self, in_channels, out_channels, channels_hidden=64, kernel_size=3, leaky_slope=0.01):
        super().__init__()

        pad = kernel_size // 2

        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(in_channels, channels_hidden,                   kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(in_channels + channels_hidden, channels_hidden, kernel_size=kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(in_channels + 2*channels_hidden, out_channels,  kernel_size=1, padding=0)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), self.leaky_slope)
        x2 = F.leaky_relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.conv3(torch.cat([x, x1, x2], 1))
        return x3

class F_fully_shallow(nn.Module):

    def __init__(self, size_in, size, internal_size = None, dropout=0.0):
        super().__init__()
        if not internal_size:
            internal_size = 2*size

        self.d1 =  nn.Dropout(p=dropout)
        self.d2 =  nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc2 = nn.Linear(internal_size, internal_size)
        self.fc3 = nn.Linear(internal_size, size)

        self.nl1 = nn.LeakyReLU()
        self.nl2 = nn.LeakyReLU()

    def forward(self, x):
        out = self.nl1(self.d1(self.fc1(x)))
        out = self.nl2(self.d2(self.fc2(out)))
        out = self.fc3(out)
        return out
