import unittest
import torch
from signal_models.priors.base_prior import BasePrior
from signal_models.priors.concat_prior import ConcatPrior
from typing import List

class MockPrior(BasePrior):
    def __init__(self, value):
        self.value = value

    def sample(self, n_samples):
        return torch.full((n_samples,1), self.value)

    def prior_fim(self):
        return torch.tensor([[self.value]])

    def prior_score(self, p):
        return torch.tensor([self.value])

class TestConcatPrior(unittest.TestCase):
    def setUp(self):
        self.prior1 = MockPrior(1)
        self.prior2 = MockPrior(2)
        self.concat_prior = ConcatPrior([self.prior1, self.prior2])

    def test_sample(self):
        samples = self.concat_prior.sample(3)
        expected = torch.tensor([[1, 2], [1, 2], [1, 2]])
        self.assertTrue(torch.equal(samples, expected))

    def test_prior_fim(self):
        fim = self.concat_prior.prior_fim()
        expected = torch.tensor([[1, 0], [0, 2]])
        self.assertTrue(torch.equal(fim, expected))

    def test_prior_score(self):
        score = self.concat_prior.prior_score(0)
        expected = torch.tensor([[1, 2]])
        self.assertTrue(torch.equal(score, expected))

if __name__ == '__main__':
    unittest.main()