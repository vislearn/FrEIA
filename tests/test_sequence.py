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

    def test_replace_layer(self):
        dim = 2
        inn = SequenceINN(dim)
        for _ in range(3):
            inn.append(AllInOneBlock(inn.output_dims(), subnet_constructor=subnet))

        new_block = AllInOneBlock([inn.shapes[1]], subnet_constructor=subnet)
        inn[1] = new_block
        self.assertTrue(inn[1] is new_block)

        # Wrong dimension
        with self.assertRaises(ValueError):
            inn[1] = AllInOneBlock([(dim + 1,)], subnet_constructor=subnet)

        # Block suddenly got condition
        with self.assertRaises(ValueError):
            inn[1] = AllInOneBlock([(dim,)], dims_c=[(dim,)], subnet_constructor=subnet)
