import matplotlib.pyplot as plt
import torch

from signal_models.priors.base_prior import BasePrior
import numpy as np

PI = np.pi


class RollingUniform(BasePrior):
    def __init__(self, a, b, epsilon, m, alpha=3):
        super().__init__()
        self.a = a * self.one
        self.b = b * self.one
        self.epsilon = epsilon * self.one
        self.alpha = alpha
        self.m = m
        if self.m != 1:
            raise ValueError("m must be 1")
        if self.a >= self.b:
            raise ValueError("a must be smaller than b")
        if self.alpha <= 1:
            raise ValueError("alpha must be greater than 1")
        self.u = (self.b - self.a - 2 * self.epsilon) + 2 * self.epsilon / (alpha + 1)

    def pdf(self, x):
        index_mid = torch.logical_and(x <= self.b - self.epsilon, x >= self.a + self.epsilon)
        index_left = x < self.a + self.epsilon
        index_right = x > self.b - self.epsilon
        pdf = torch.zeros_like(x)
        pdf[index_right] = torch.pow((self.b - x[index_right]) / self.epsilon, self.alpha)
        pdf[index_mid] = 1
        pdf[index_left] = torch.pow((x[index_left] - self.a) / self.epsilon, self.alpha)
        return pdf / self.u

    def inv_cdf(self, p):

        epsilon_cdf = self.epsilon / ((self.alpha + 1) * self.u)
        index_left = p < epsilon_cdf
        index_right = p > 1 - epsilon_cdf
        index_mid = torch.logical_and(p >= epsilon_cdf, p <= 1 - epsilon_cdf)
        inv_cdf = torch.zeros_like(p)

        p = p * self.u
        inv_cdf[index_left] = self.a + self.epsilon * (p[index_left] * (self.alpha + 1) / self.epsilon) ** (
                1 / (self.alpha + 1))
        inv_cdf[index_mid] = self.a + self.epsilon + p[index_mid] - (self.epsilon / (self.alpha + 1))

        z = (self.alpha + 1) * (self.b - self.a - 2 * self.epsilon + 2 * self.epsilon / (self.alpha + 1) - p[
            index_right]) / self.epsilon
        inv_cdf[index_right] = self.b - self.epsilon * z ** (1 / (self.alpha + 1))
        return inv_cdf

    def prior_score(self, x):
        index_mid = torch.logical_and(x <= self.b - self.epsilon, x >= self.a + self.epsilon)
        index_left = x < self.a + self.epsilon
        index_right = x > self.b - self.epsilon
        score = torch.zeros_like(x)
        score[index_mid] = 0
        score[index_left] = self.alpha * self.epsilon / (x[index_left] - self.a)
        score[index_right] = -self.alpha * self.epsilon / (self.b - x[index_right])
        return score.reshape([-1, self.m])

    def sample(self, n_samples):
        p = torch.rand([n_samples], device=self.device)
        return self.inv_cdf(p).reshape([-1, self.m])

    def prior_fim(self):
        return 2 * self.alpha ** 2 * self.epsilon / ((self.alpha - 1) * self.u)

    def plot(self):
        import matplotlib.pyplot as plt
        x = torch.linspace(self.a[0], self.b[0], 1000, device=self.device)
        plt.plot(x.cpu().numpy(), self.pdf(x).cpu().numpy())


if __name__ == '__main__':
    import numpy as np

    ru = RollingUniform(-np.pi, np.pi, 0.1 * np.pi, 1)
    print(ru.prior_fim())
    x = ru.sample(200000)
    s = ru.prior_score(x)
    print(torch.mean(s ** 2))
    ru.plot()
    # plt.show()
    plt.hist(x.cpu().numpy().flatten(), bins=100, density=True, color="red")
    plt.show()
