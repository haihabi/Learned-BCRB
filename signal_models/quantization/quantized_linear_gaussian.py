from enum import Enum
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import pyresearchutils as pru
from torch import nn
from torch import distributions

from signal_models.base_problem import BaseProblem, compute_problem_fims
from signal_models.quantization.base_quantization import PriorConfig
from signal_models.quantization.n_bits_quantization import NBitQuantization
from signal_models.quantization.one_bit_quantization import OneBitQuantization

PI_SQRT = np.sqrt(np.pi)


class QuantizedLinearProblem(BaseProblem):
    """
    The class for the quantized linear problem.
    """

    def __init__(self, n, m, k, sigma_prior=0.5, rho_cov: float = 0, n_bits: int = 1, minimal_snr=-3.5, maximal_snr=1.5,
                 snr_points=20, *args, **kwargs):
        """
        Initialize the QuantizedLinearProblem class.
        :param n:  The number of measurements.
        :param m:  The number of parameters.
        :param k:  The number of i.i.d. samples.
        :param sigma_prior:  The prior sigma.
        :param rho_cov:  The covariance rho.
        :param n_bits:  The number of bits for quantization.
        """
        snr_array = torch.logspace(minimal_snr, maximal_snr, snr_points).to(pru.get_working_device())
        condition_list = sigma_prior / torch.sqrt(snr_array)
        super().__init__(n, m, k, condition_list, has_bcrb=False, is_complex=False,has_score=True if rho_cov==0 else False)

        w_matrix = torch.randn([n, m]).to(pru.get_working_device())
        w_matrix = w_matrix / torch.norm(w_matrix)
        self.register_parameter("w_matrix", nn.Parameter(w_matrix, requires_grad=False))
        self.rho = rho_cov

        if n_bits == 1:
            self.obq = OneBitQuantization(self.phi, self.dphidtheta)
        else:
            self.obq = NBitQuantization(n_bits, self.phi, self.dphidtheta)
        self.set_sigma_prior(sigma_prior)

    def phi(self, in_theta):
        """
        Generate a signal from the given parameters.
        :param in_theta:  The parameter theta. The shape is [B, N].
        :param in_w_matrix:
        :return:
        """
        return in_theta @ self.w_matrix.T + 0.75

    def dphidtheta(self, p):
        """
        Compute the derivative of the signal with respect to the parameters.
        :param in_theta: The parameter theta.
        :return: The derivative of the signal with respect to the parameters.
        """
        return self.w_matrix.unsqueeze(dim=0).unsqueeze(
            dim=0)  # The derivative of the signal with respect to the parameters.

    def set_sigma_prior(self, sigma_prior):
        """
        Set the prior sigma.
        :param sigma_prior:
        :return: None
        """
        self.sigma_prior = sigma_prior
        self.obq.set_prior(PriorConfig(scale=sigma_prior))

    def get_optimal_likelihood_score(self) -> Callable:
        def optimal_score_function(in_x, in_theta, in_sigma):
            if self.rho != 0:
                return None  # The optimal score is not available in cases where the rho is not zero.
            score_likelihood = self.obq.likelihood_score_function(in_x, in_theta, in_sigma)
            return score_likelihood

        return optimal_score_function

    def get_prior_fim(self):
        return self.obq.prior_fim()

    def get_optimal_prior_score(self) -> Callable:
        def optimal_score_function(in_theta):
            return -in_theta / self.sigma_prior ** 2

        return optimal_score_function

    def get_dataset(self, dataset_size, cond=None, iid_dataset=False):
        r = (torch.ones(self.n, self.n) - torch.eye(self.n)) * self.rho
        _noise = distributions.MultivariateNormal(torch.zeros(self.n).to(self.device),
                                                  covariance_matrix=(torch.eye(self.n) + r).to(self.device))

        p = self.obq.sample_parameters(dataset_size, self.m).to(pru.get_working_device()).float()
        if iid_dataset:
            noise = _noise.sample([dataset_size, self.k]).to(self.device)
        else:
            noise = _noise.sample([dataset_size]).to(self.device)
        sigma_noise_array = self.get_condition_list()
        if cond is None:
            index = torch.randint(low=0, high=len(sigma_noise_array), size=(dataset_size, 1)).flatten()
            cond = sigma_noise_array[index].to(self.device)
        else:
            cond = torch.ones(dataset_size).to(self.device) * cond

        signal = self.obq.generate_signal(p).squeeze(1)
        if iid_dataset:
            signal = signal.unsqueeze(-2)  # Add the i.i.d. dimension
            noise = noise * cond.reshape([dataset_size, 1, 1])  # Add the i.i.d. dimension to the noise
        else:
            noise = noise * cond.unsqueeze(-1)

        measurement = self.obq.quantization(signal + noise)  # Apply quantization to the measurements

        return pru.NumpyDataset(measurement.cpu().numpy().astype("float32"),
                                p.cpu().numpy().astype("float32"),
                                metadata=cond.cpu().numpy(),
                                transform=None)


if __name__ == '__main__':
    p = QuantizedLinearProblem(32, 1, 16, sigma_prior=0.25, rho_cov=0, n_bits=1)
    p.set_sigma_prior(0.25)
    res = []
    for snr in p.get_condition_list():
        fim, pfim = compute_problem_fims(p, snr)
        bcrb = torch.linalg.inv(pfim + fim)
        res.append(bcrb.item())
    plt.plot(res)
    plt.show()
    # print("a")
    # score_likelihood = p.get_optimal_likelihood_score()
    # score_prior = p.get_optimal_prior_score()
