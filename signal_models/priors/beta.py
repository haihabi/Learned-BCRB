import numpy as np
import torch

from signal_models.priors.base_prior import BasePrior
from torch import distributions


def compute_prior_crb(alpha, beta, delta):
    p1 = (alpha + beta - 1) * (alpha + beta - 2) / delta ** 2
    p21 = 1 / (alpha - 2)
    p22 = 1 / (beta - 2)
    return p1 * (p21 + p22)


def beta_score(x, alpha, beta, a, b):
    return (alpha - 1) / (x - a) - (beta - 1) / (b - x)


class BetaPrior(BasePrior):
    def __init__(self, alpha, beta, a, b, m):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.m = m
        if self.a >= self.b:
            raise ValueError("a must be smaller than b")

        self.prior_dist = distributions.TransformedDistribution(
            distributions.Beta(self.alpha * self.one, self.beta * self.one),
            distributions.AffineTransform(loc=self.a * self.one,
                                          scale=(self.b - self.a) * self.one))

    def sample(self, n_samples):
        return self.prior_dist.sample([n_samples, self.m]).to(self.device).reshape([-1, self.m])

    def prior_fim(self):
        with torch.no_grad():
            prior_fim = torch.eye(self.m, device=self.device)
            for i in range(self.m):
                prior_fim[i, i] = compute_prior_crb(self.alpha, self.beta, self.b - self.a)
            return prior_fim

    def prior_score(self, p):
        return beta_score(p, self.alpha, self.beta, self.a, self.b).reshape([-1, self.m])

    def variance(self):
        base_var = self.alpha * self.beta / (((self.alpha + self.beta) ** 2) * (self.alpha + self.beta + 1))
        return base_var * (self.b - self.a) ** 2

    def mean(self):
        return (self.alpha / (self.alpha + self.beta)) * (self.b - self.a) + self.a

    def power(self):
        return self.variance() + self.mean() ** 2

    def plot(self):
        import matplotlib.pyplot as plt
        x = torch.linspace(self.a+0.001, self.b-0.001, 1000, device=self.device)
        y = torch.exp(self.prior_dist.log_prob(x))
        plt.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy())
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("p(x)")
        plt.tight_layout()
        # plt.show()




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    b = BetaPrior(2.1, 2.1, -np.pi, np.pi, 1)

    b.plot()
    plt.show()