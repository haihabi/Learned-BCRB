import math
from typing import Callable

import torch
from torch import Tensor

from signal_models.quantization.base_quantization import PriorConfig, gaussian_cdf


class OneBitQuantization:
    def __init__(self, in_generate_signal_function: Callable,
                 in_signal_derivative: Callable,
                 in_eps=1e-8):
        """
        Initialize the OneBitQuantization class.

        Parameters:
        in_generate_signal_function (Callable): The function to generate the signal.
        in_signal_derivative (Callable): The function to compute the derivative of the signal with respect to the parameters.
        in_eps (float): The epsilon value. Default is 1e-8.
        """

        def generate_signal(in_theta):
            return in_generate_signal_function(in_theta).unsqueeze(dim=1)

        self.generate_signal = generate_signal
        self.signal_derivative = in_signal_derivative
        self.prior = None
        self.eps = in_eps

    def quantization(self, in_x: Tensor) -> Tensor:
        """
        Perform one-bit quantization on the input tensor.

        Parameters:
        in_x (Tensor): The input tensor.

        Returns:
        Tensor: The quantized tensor.
        """
        return torch.sign(in_x)

    def set_prior(self, in_prior: PriorConfig):
        """
        Set the prior for the parameters.
        :param in_prior:
        :return:
        """
        self.prior = in_prior

    def sample_parameters(self, in_batch_size: int, in_m: int) -> Tensor:
        """
        Sample the parameters.

        Parameters:
        in_batch_size (int): The batch size.
        in_m (int): The number of parameters.

        Returns:
        Tensor: The sampled parameters.
        """
        return self.prior.sample([in_batch_size, in_m])

    def prior_fim(self) -> Tensor:
        """
        Compute the Fisher Information Matrix for the prior.
        :return:
        """
        return self.prior.prior_fim()

    def fisher_one_bit_linear(self,
                              in_theta: Tensor,
                              in_sigma_array: Tensor) -> Tensor:
        """
        Compute the Fisher Information Matrix for one-bit linear quantization.

        Parameters:
        in_theta (Tensor): The parameter theta. The shape is [B, N].
        in_sigma_array (Tensor): The array of sigma values. The shape is [M].


        Returns:
        Tensor: The Fisher Information Matrix.
        """
        _sigma_array = in_sigma_array.reshape([1, -1, 1])

        def phi(x):
            return gaussian_cdf(x, torch.zeros(1, device=x.device).reshape([1]), _sigma_array)

        s_n = self.generate_signal(in_theta)
        ds_n_dtheta = self.signal_derivative(in_theta)
        mat_dsn_dtheta = (ds_n_dtheta.unsqueeze(dim=-1) @ ds_n_dtheta.unsqueeze(dim=-2))

        exp_term = s_n ** 2 / (_sigma_array ** 2)
        div_phi_sn = phi(s_n) * phi(-s_n)
        q_scale = torch.exp(-exp_term) / (
                div_phi_sn * 2 * math.pi * _sigma_array ** 2)
        q_scale[div_phi_sn == 0] = 0
        q_scale = q_scale.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return torch.sum(mat_dsn_dtheta * q_scale, dim=2)

    def likelihood_score_function(self, in_x, in_theta, in_sigma_array):
        """
        Compute the likelihood score function for the given input.

        :param in_x: The input tensor.
        :param in_theta: The parameter theta.
        :param in_sigma_array: The sigma value.

        :return: The score function.
        """
        _sigma_array = in_sigma_array.reshape([-1, 1, 1])

        def phi(x):
            return gaussian_cdf(x, torch.zeros(1, device=x.device).reshape([1]), _sigma_array)

        s_n = self.generate_signal(in_theta)
        base = phi(-s_n)
        pos = (in_x == 1).float()
        neg = 1 - pos

        one_over_likelihood = 1 / ((1 - base) * pos + neg * base + 1e-8)
        dtheta_dlikelihood = (pos - neg) / (math.sqrt(2 * math.pi) * _sigma_array) * torch.exp(
            -s_n ** 2 / (2 * _sigma_array ** 2)) * self.signal_derivative(in_theta).reshape([1, -1])

        return torch.sum(torch.sum(dtheta_dlikelihood * one_over_likelihood, dim=-1), dim=-1, keepdim=True)

    def map_estimator(self,
                      in_x, in_sigma_array, in_m, eps=1e-8):
        """
        Compute the MAP estimator for the given input.




        :param in_x: The input tensor [B, S, K, N] with K i.i.d samples .
        :param in_sigma_array: The array of sigma values of size S.
        :param in_m: The number of parameters.
        :param eps: The epsilon value. Default is 1e-8.

        :return: The MAP estimate.
        """
        _sigma_array = in_sigma_array.reshape([1, -1, 1, 1]).double()

        def phi(x):
            return gaussian_cdf(x, torch.zeros(1, device=x.device).reshape([1]), _sigma_array)

        def neg_log_likelihood(in_theta):
            s_n = self.generate_signal(in_theta)
            base = phi(-s_n)
            pos = (in_x == 1).double()
            neg = 1 - pos

            return -torch.sum(torch.sum(torch.log((1 - base) * pos + neg * base + eps), dim=-1), dim=-1)

        def neg_log_posterior(in_theta):
            return neg_log_likelihood(in_theta) + self.prior.neg_log_prior(in_theta).squeeze(
                dim=-1)  # The squeeze is for remove the i.i.d samples dim which is always 1.

        from torch import nn
        param_init = self.prior.sample([*in_x.shape[:-2], 1, in_m]).to(in_x.device)  # The one is for the i.i.d samples
        param_opt = nn.Parameter(param_init,
                                 requires_grad=True)
        # Create an optimizer
        optimizer = torch.optim.Adam([param_opt], lr=0.1)

        # Run the optimization
        loss_array = []
        for _ in range(2000):
            optimizer.zero_grad()
            loss = torch.mean(neg_log_posterior(param_opt))
            loss.backward()
            loss_array.append(loss.item())
            optimizer.step()
            param_opt.data += torch.randn_like(param_opt) * 0.0
            param_opt.data = self.prior.parameter_projection(param_opt.data)

        return param_opt.squeeze(
            dim=-2).clone().detach()  # The squeeze is for remove the i.i.d samples dim which is always 1.
