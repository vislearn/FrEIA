import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class F_conv(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=None,
                 stride=None, kernel_size=3, leaky_slope=0.1,
                 batch_norm=False):
        super().__init__()

        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(in_channels, channels_hidden,
                               kernel_size=kernel_size, padding=pad,
                               bias=not batch_norm)
        self.conv2 = nn.Conv2d(channels_hidden, channels_hidden,
                               kernel_size=kernel_size, padding=pad,
                               bias=not batch_norm)
        self.conv3 = nn.Conv2d(channels_hidden, channels,
                               kernel_size=kernel_size, padding=pad,
                               bias=not batch_norm)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(channels_hidden)
            self.bn1.weight.data.fill_(1)
            self.bn2 = nn.BatchNorm2d(channels_hidden)
            self.bn2.weight.data.fill_(1)
            self.bn3 = nn.BatchNorm2d(channels)
            self.bn3.weight.data.fill_(1)
        self.batch_norm = batch_norm

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.leaky_relu(out, self.leaky_slope)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = F.leaky_relu(out, self.leaky_slope)

        out = self.conv3(out)
        if self.batch_norm:
            out = self.bn3(out)
        return out


class F_fully_connected(nn.Module):
    '''Fully connected tranformation, not reversible, but used below.'''

    def __init__(self, size_in, size, internal_size=None, dropout=0.0):
        super().__init__()
        if not internal_size:
            internal_size = 2*size

        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d2b = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc2 = nn.Linear(internal_size, internal_size)
        self.fc2b = nn.Linear(internal_size, internal_size)
        self.fc3 = nn.Linear(internal_size, size)

        self.nl1 = nn.ReLU()
        self.nl2 = nn.ReLU()
        self.nl2b = nn.ReLU()

    def forward(self, x):
        out = self.nl1(self.d1(self.fc1(x)))
        out = self.nl2(self.d2(self.fc2(out)))
        out = self.nl2b(self.d2b(self.fc2b(out)))
        out = self.fc3(out)
        return out

class F_fully_convolutional(nn.Module):

    def __init__(self, in_channels, out_channels, internal_size=256, kernel_size=3, leaky_slope=0.02):
        super().__init__()

        pad = kernel_size // 2

        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(in_channels, internal_size,                  kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(in_channels + internal_size, internal_size,  kernel_size=kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(in_channels + 2*internal_size, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), self.leaky_slope)
        x2 = F.leaky_relu(self.conv2(torch.cat([x, x1], 1)), self.leaky_slope)
        return self.conv3(torch.cat([x, x1, x2], 1))
