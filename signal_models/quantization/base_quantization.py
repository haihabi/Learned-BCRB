import math
from typing import List

import torch
from torch import Tensor
from torch.distributions import Beta
import pyresearchutils as pru
from signal_models.base_problem import PriorType


class PriorConfig:
    def __init__(self,
                 scale: float = 1,
                 mean: float = 0,
                 prior_type: PriorType = PriorType.GAUSSIAN,
                 alpha=4.2,
                 beta=4.2,
                 device="cpu"):
        """
        Initialize the PriorConfig class.
        :param scale: The scale of the prior. Default is 1.
        :param mean: The mean of the prior. Default is 0.
        :param prior_type: The type of the prior. Default is PriorType.GAUSSIAN.
        """
        self.scale = scale
        self.mean = mean
        self.prior_type = prior_type
        self.beta_dist = Beta(torch.tensor([alpha], device=device), torch.tensor([beta], device=device))
        self.alpha = alpha
        self.beta = beta
        self.device = pru.get_working_device()


    def sample(self, in_shape: List[int], device: str = "cpu") -> Tensor:
        """
        Sample the prior.
        :param in_shape:
        :param device:
        :return:
        """
        if self.prior_type == PriorType.GAUSSIAN:
            return self.scale * torch.randn(in_shape, device=device) + self.mean
        elif self.prior_type == PriorType.BETA:
            return self.beta_dist.sample(in_shape).to(device).squeeze(dim=-1) * self.scale + self.mean
            # return torch.rand(in_shape, device=device) * self.scale + self.mean
        else:
            raise Exception("Unknown prior type")

    def parameter_projection(self, in_theta: Tensor) -> Tensor:
        """
        Project the parameter to the valid range.
        :param in_theta:
        :return:
        """
        if self.prior_type == PriorType.GAUSSIAN:
            return in_theta
        elif self.prior_type == PriorType.BETA:
            max_value = self.scale + self.mean - 1e-3
            theta_clip = torch.clamp(in_theta, 0, max_value)
            reflect = 2 * theta_clip - in_theta
            return reflect

        else:
            raise Exception("Unknown prior type")

    def neg_log_prior(self, in_theta: Tensor) -> Tensor:
        """
        Compute the negative log prior.
        :param in_theta:
        :return:
        """
        if self.prior_type == PriorType.GAUSSIAN:
            return torch.linalg.norm(in_theta, dim=-1) ** 2 / (2 * self.scale ** 2)
        elif self.prior_type == PriorType.BETA:
            # self.beta.to(in_theta.device)
            return -self.beta_dist.log_prob(in_theta / self.scale).squeeze(dim=-1)
        else:
            raise Exception("Unknown prior type")

    def prior_fim(self) -> float:
        """
        Compute the Fisher Information Matrix for the prior.
        :param in_theta:
        :return:
        """
        if self.prior_type == PriorType.GAUSSIAN:
            return torch.tensor([1 / self.scale ** 2], device=self.device).reshape([1, 1])
        elif self.prior_type == PriorType.BETA:
            a_h = self.scale + self.mean
            a_l = self.mean
            distance_factor = 1 / (a_h - a_l) ** 2
            return (self.beta + self.alpha - 1) * (self.beta + self.alpha - 2) * distance_factor * (
                    1 / (self.alpha - 2) + 1 / (self.beta - 2))
        else:
            raise Exception("Unknown prior type")


def gaussian_cdf(in_x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """
    Compute the Gaussian cumulative distribution function (CDF).

    Parameters:
    x (Tensor): The input tensor.
    mu (Tensor): The mean of the Gaussian distribution.
    sigma (Tensor): The standard deviation of the Gaussian distribution.

    Returns:
    Tensor: The CDF of the Gaussian distribution at x.
    """
    return 0.5 * (1 + torch.erf((in_x - mu) / (sigma * SQRT2)))


SQRT2 = math.sqrt(2)
SQRT2PI = math.sqrt(2 * math.pi)


def infmean(in_x, in_dim):
    """
    Compute the mean of the input tensor ignoring the inf values.
    :param in_x:  The input tensor
    :param in_dim:
    :return:
    """
    res = []
    for i in range(in_x.shape[in_dim]):
        _x = torch.select(in_x, in_dim, i)
        m = torch.mean(_x[~torch.isinf(_x)])
        res.append(m)
    return torch.stack(res)
