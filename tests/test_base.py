import unittest

import torch
import torch.nn as nn
import torch.optim
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import sys
sys.path.append('../')

from FrEIA.modules import InvertibleModule


class BaseLayerTest(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)

        self.batch_size = 32
        self.tol = 1e-4

        input_size = 8
        cond_size = 32 #this could be 2D too

        torch.manual_seed(self.batch_size)

        self.x = torch.randn(self.batch_size, *input_size).to(DEVICE)
        self.c = torch.randn(self.batch_size, *cond_size).to(DEVICE)

    def test_constructs(self):

        b = InvertibleModule(self.x.shape, self.c.shape)
        self.assertTrue(isinstance(b, InvertibleModule))
