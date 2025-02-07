import torch

from signal_models.priors.base_prior import BasePrior


class GaussianPrior(BasePrior):
    def __init__(self, mean, var, m):
        super().__init__()
        self.mean = mean
        self.var = var * self.one
        self.m = m

    def prior_fim(self):
        return (1 / self.var) * torch.eye(self.m, device=self.device)

    def sample(self, n_samples):
        return torch.randn(n_samples, self.m, device=self.device) * torch.sqrt(self.var) + self.mean

    def prior_score(self, p):
        return -((p - self.mean) / self.var).reshape([-1, self.m])


    def power(self):
        return self.var+ self.mean ** 2