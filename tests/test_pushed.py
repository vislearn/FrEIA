import unittest

import torch

from FrEIA.distributions import StandardNormalDistribution, PullBackDistribution, PushForwardDistribution
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock


def subnet(dim_in, dim_out):
    return torch.nn.Sequential(
        torch.nn.Linear(dim_in, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, dim_out)
    )


class PushedDistributionTest(unittest.TestCase):
    def create_pull_back_distribution(self):
        inn = SequenceINN(2)
        inn.append(AllInOneBlock((inn.shapes[-1],), subnet_constructor=subnet))
        latent = StandardNormalDistribution(2)
        distribution = PullBackDistribution(latent, inn)
        return distribution

    def create_push_forward_distribution(self):
        inn = SequenceINN(2)
        inn.append(AllInOneBlock((inn.shapes[-1],), subnet_constructor=subnet))
        latent = StandardNormalDistribution(2)
        distribution = PushForwardDistribution(latent, inn)
        return distribution

    def test_log_prob(self):
        self.create_pull_back_distribution().log_prob(torch.randn(16, 2))
        self.create_push_forward_distribution().log_prob(torch.randn(16, 2))

    def test_log_prob_shape_mismatch(self):
        with self.assertRaises(RuntimeError):
            self.create_pull_back_distribution().log_prob(torch.randn(16, 3))
        with self.assertRaises(RuntimeError):
            self.create_push_forward_distribution().log_prob(torch.randn(16, 3))

    def test_sample(self):
        batch_size = 16

        sample = self.create_pull_back_distribution().sample((batch_size,))
        self.assertFalse(sample.requires_grad)
        self.assertEqual(sample.shape, (batch_size, 2))

        sample = self.create_push_forward_distribution().sample((batch_size,))
        self.assertFalse(sample.requires_grad)
        self.assertEqual(sample.shape, (batch_size, 2))
