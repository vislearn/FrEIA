import unittest

import torch.nn

from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock


def subnet(dim_in, dim_out):
    return torch.nn.Sequential(
        torch.nn.Linear(dim_in, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, dim_out)
    )


class SequenceINNTest(unittest.TestCase):
    def test_append_class(self):
        input_shape = (2,)
        inn = SequenceINN(*input_shape)
        inn.append(AllInOneBlock, subnet_constructor=subnet)
        self.assertEqual(inn.shapes[-1], input_shape)

    def test_append_instance(self):
        input_shape = (2,)
        inn = SequenceINN(*input_shape)
        inn.append(AllInOneBlock(inn.output_dims(), subnet_constructor=subnet))
        self.assertEqual(inn.shapes[-1], input_shape)
