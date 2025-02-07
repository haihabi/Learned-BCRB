import math
from typing import Callable

import torch
import pyresearchutils as pru
from torch import Tensor
from signal_models.quantization.base_quantization import gaussian_cdf, SQRT2, SQRT2PI
from signal_models.quantization.one_bit_quantization import OneBitQuantization


class NBitQuantization(OneBitQuantization):
    def __init__(self, n_bits: int, in_generate_signal_function: Callable, in_signal_derivative: Callable):
        """
        Initialize the OneBitQuantization class.

        Parameters:
        n_bits (int): The number of bits for quantization.
        in_generate_signal_function (Callable): The function to generate the signal.
        in_signal_derivative (Callable): The function to compute the derivative of the signal with respect to the parameters.
        """
        super().__init__(in_generate_signal_function, in_signal_derivative)
        self.n_bits = n_bits
        self.threshold = 1
        if n_bits < 2:
            raise ValueError("The number of bits should be at least 2.")
        self.device = pru.get_working_device()
        self.delta = 2 * self.threshold / (2 ** n_bits - 1)
        self.b_array = -self.threshold + self.delta * torch.arange(2 ** n_bits + 1).to(self.device) - self.delta / 2
        self.b_array[0] = -torch.inf
        self.b_array[-1] = torch.inf
        self.q_array = -self.threshold + self.delta * torch.arange(2 ** n_bits).to(self.device)

    def quantization(self, in_x: Tensor) -> Tensor:
        """
        Perform n-bit quantization on the input tensor.

        Parameters:
        in_x (Tensor): The input tensor.

        Returns:
        Tensor: The quantized tensor.
        """

        return torch.clamp(self.delta * (torch.floor(in_x / self.delta) + 0.5), -self.threshold, self.threshold)

    def quantization_boundaries(self, in_x: Tensor) -> (Tensor, Tensor):
        """
        Compute the quantization boundaries for the given input.

        Parameters:
        in_x (Tensor): The input tensor.

        Returns:
        (Tensor, Tensor): The lower and upper quantization boundaries.
        """
        _x = in_x.reshape([-1])
        index = torch.abs(self.q_array.reshape([-1, 1]) - _x.unsqueeze(0)).argmin(dim=0)
        b_l = self.b_array[index]
        b_h = self.b_array[index + 1]
        return b_l.reshape(in_x.shape), b_h.reshape(in_x.shape)

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
        raise NotImplementedError("Implement the rest of the function")
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

        b_l, b_h = self.quantization_boundaries(in_x)

        delta_l = (b_l - s_n) / (_sigma_array * SQRT2)
        delta_h = (b_h - s_n) / (_sigma_array * SQRT2)

        delta_exp = torch.exp(-delta_l ** 2) - torch.exp(-delta_h ** 2)
        delta_phi = phi(b_h - s_n) - phi(b_l - s_n) + self.eps

        scale = 1 / (SQRT2PI * _sigma_array)

        w = (scale * delta_exp / delta_phi).squeeze(1)
        dmudtheta = self.signal_derivative(in_theta).reshape([1, 1, -1])
        _score = dmudtheta * w
        score_likelihood = torch.sum(_score, dim=(1, 2)).unsqueeze(1)

        score = score_likelihood
        if torch.any(torch.isnan(score)) or torch.any(torch.isinf(score)):
            print("Nan in score")
        return score

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
            b_l, b_h = self.quantization_boundaries(s_n)
            delta_phi = phi(b_h - s_n) - phi(b_l - s_n)

            return -torch.sum(torch.sum(torch.log(delta_phi + eps), dim=-1), dim=-1)

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
