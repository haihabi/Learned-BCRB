import torch
import pyresearchutils as pru
from torch import distributions
from typing import Callable, Mapping, Any
import scipy

from signal_models.base_problem import BaseProblem
import numpy as np

from signal_models.frequency_estimation.frequency_estimation_utils import get_alpha_beta_values, load_wav_dataset, \
    compute_covariance_matrix
from signal_models.priors.beta import BetaPrior
from signal_models.priors.concat_prior import ConcatPrior
from signal_models.priors.rolling_uniform import RollingUniform
from utils.config import NoiseType


class FrequencyEstimationMultitone(BaseProblem):
    def __init__(self, n, m, k, alpha_rho, beta_rho, minimal_snr=-4, maximal_snr=1, snr_points=20,
                 noise_type: NoiseType = NoiseType.GaussianScale,
                 center_frequency=0.5 * np.pi,
                 spacing=0.1 * np.pi,
                 is_random_amplitude=False,
                 is_random_phase=False,
                 in_covariance=None,
                 *args, **kwargs):
        """
        This class implements the linear model with a Gaussian prior.
        :param n:  Number of measurements (int)
        :param m:  Number of parameters (int)
        :param k:  Number of i.i.d. samples (int)
        :param alpha_rho:  Shape parameter of the Gamma distribution (float)
        :param beta_rho: Rate parameter of the Gamma distribution (float)
        :param minimal_snr: Minimal SNR in dB (float)
        :param maximal_snr: Maximal SNR in dB (float)
        :param snr_points: Number of SNR points (int)
        :param noise_type: Type of noise (NoiseType)
        :param center_frequency: Center frequency of the signal (float)
        :param spacing: Spacing between the tones (float)
        :param is_random_amplitude: If True, the amplitude is random (bool)
        :param is_random_phase: If True, the phase is random (bool)
        :param in_covariance: Covariance matrix of the noise (torch.Tensor)
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        """
        condition_list = 10 * torch.log10(torch.logspace(minimal_snr, maximal_snr, snr_points))
        super().__init__(n, m, k, condition_list, has_bcrb=True if noise_type == NoiseType.GaussianScale else False,
                         is_complex=False)

        self.n = n
        self.m = m
        self.k = k
        self.spacing = spacing
        if self.m != 1 + int(is_random_phase):
            raise ValueError(f"The number of parameters must be {1 + int(is_random_phase)}")
        self.t = torch.linspace(0, n - 1, n).to(self.device)
        self.is_random_phase = is_random_phase

        v = np.pi ** 2 / (4 * (2 * alpha_rho + 1))
        alpha, beta = get_alpha_beta_values(center_frequency, v, 0, np.pi)
        self.alpha_rho = alpha
        param_list = []
        param_list.append(BetaPrior(alpha, beta, 0, np.pi, 1))
        if is_random_amplitude:
            param_list.append(BetaPrior(0.8, 1.2, self.alpha_rho, self.alpha_rho))
        if is_random_phase:
            # param_list.append(RollingUniform(-np.pi, np.pi, 0.1 * np.pi, 1))
            param_list.append(BetaPrior(3, 3,-np.pi, np.pi, 1))
        self.prior = ConcatPrior(param_list)
        self.noise_type = noise_type
        if noise_type == NoiseType.Real:
            cov = torch.eye(n).to(self.device)
        elif noise_type == NoiseType.GaussianScale:
            cov = torch.eye(n).to(self.device)
        elif noise_type == NoiseType.GaussianMatrix:
            if self.k != 1:
                raise ValueError("Only k=1 is supported for Gaussian noise")
            if in_covariance is None:
                noise = self._samples_noise(64000, True, noise_type=NoiseType.Real)
                remove_mean = noise - torch.mean(noise, dim=0, keepdim=True)
                cov = torch.mean(torch.permute(remove_mean, dims=[0, 2, 1]) @ remove_mean, dim=0)
            else:
                cov = in_covariance

        else:
            raise ValueError("Invalid noise type")
        self.register_buffer("cov", cov)
        self.register_buffer("cov_inv", torch.linalg.inv(cov))

        self.recompute_snr_base()

    def change_real2gaussian(self, in_type=NoiseType.GaussianMatrix):
        self.noise_type = in_type
        if in_type == NoiseType.Real:
            raise ValueError("Real noise is not supported")
        elif in_type == NoiseType.GaussianScale:
            cov = torch.eye(self.n).to(self.device) * torch.diag(self.cov).mean()
            self.cov.data = cov
        self.cov_inv.data = torch.linalg.inv(self.cov)
        self.recompute_snr_base()

    def get_prior_fim(self):
        return self.prior.prior_fim()

    def phi(self, p):
        if self.is_random_phase:
            return torch.cos(p[:, 0].reshape([-1, 1]) * self.t.reshape([1, -1]) + p[:, 1].reshape([-1, 1]))
        return torch.cos(p[:, 0].reshape([-1, 1]) * self.t.reshape([1, -1]))

    def dphidtheta(self, p):
        main_term = p[:, 0].reshape([-1, 1]) * self.t.reshape([1, -1])
        if self.is_random_phase:
            main_term += p[:, 1].reshape([-1, 1])
        dphidtheta = -(torch.sin(main_term) * self.t.reshape([1, -1])).unsqueeze(1)
        if self.is_random_phase:
            dphidtheta = torch.cat([dphidtheta, -torch.sin(main_term).unsqueeze(1)], dim=1)
        return dphidtheta

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        super().load_state_dict(state_dict, strict)
        self.recompute_snr_base()

    def recompute_snr_base(self):
        pow_theta = 0.5
        _snr = pow_theta / torch.max(torch.diag(self.cov))
        self.snr_base = _snr.item()

    def bcrb(self, snr, k_iid=1):
        if self.noise_type != NoiseType.GaussianScale:
            return None, None, None
        snr_nat = 10 ** (snr / 10)
        cov_snr_factor = self.snr_base / snr_nat
        cov = self.cov * cov_snr_factor
        one_over_sigma = torch.linalg.inv(cov)
        if self.is_random_phase:
            fim = torch.zeros([2, 2], device=self.device)
            fim[0, 0] = torch.sum(self.t ** 2) * 0.5 * one_over_sigma[0, 0]
            fim[1, 1] = self.n * 0.5 * one_over_sigma[0, 0]
            fim[0, 1] = torch.sum(self.t) * 0.5 * one_over_sigma[0, 0]
            fim[1, 0] = fim[0, 1]
        else:
            fim = torch.sum(self.t ** 2) * 0.5 * one_over_sigma[0, 0]
            fim = fim.reshape([1, 1])

        prior_fim = self.prior.prior_fim()

        return torch.linalg.inv(k_iid * fim + prior_fim), k_iid * fim, prior_fim

    def get_optimal_likelihood_score(self) -> Callable:
        def optimal_score_function(in_x, in_theta, in_snr, sum_over_iid=False):
            snr_nat = 10 ** (in_snr / 10)
            one_over_factor = (snr_nat / self.snr_base).reshape([-1, 1, 1])
            P = torch.permute(self.dphidtheta(in_theta) @ self.cov_inv, [0, 2, 1])
            delta = (in_x - self.phi(in_theta).unsqueeze(1)) @ P
            delta = one_over_factor * delta

            return torch.sum(delta, dim=1)

        return optimal_score_function

    def get_optimal_prior_score(self) -> Callable:
        def optimal_score_function(in_theta):
            return self.prior.prior_score(in_theta)

        return optimal_score_function

    def _samples_noise(self, dataset_size, iid_dataset, noise_type=None):
        noise_type = noise_type if noise_type is not None else self.noise_type
        if noise_type == NoiseType.Real:
            d_seq = load_wav_dataset(dataset_size, self.k, self.n)

            noise = torch.tensor(d_seq, device=self.device).to(self.device).float()
            noise = noise / torch.std(noise, dim=0, keepdim=True)  # Normalize the noise to have unit variance
            # Compute and save covariance matrix
            mu = torch.mean(noise, dim=0, keepdim=True)
            noise_mr = noise - mu
            cov = torch.mean(torch.permute(noise_mr, dims=[0, 2, 1]) @ noise_mr, dim=0)
            self.cov.data = cov.detach()

            return noise
        else:
            noise = distributions.MultivariateNormal(torch.zeros(self.n).to(self.device), covariance_matrix=self.cov)
            if iid_dataset:
                noise = noise.sample([dataset_size, self.k]).to(self.device)
            else:
                noise = noise.sample([dataset_size]).to(self.device)
            return noise

    def get_dataset(self, dataset_size, iid_dataset=True, cond=None):
        with torch.no_grad():
            p = self.prior.sample(dataset_size)
            noise = self._samples_noise(dataset_size, iid_dataset)

            if cond is None:
                index = torch.randint(low=0, high=len(self.get_condition_list()), size=(dataset_size, 1)).flatten()
                cond = self.get_condition_list().clone().detach()[index].to(self.device)
            else:
                cond = torch.ones(dataset_size).to(self.device) * cond
            snr_nat = 10 ** (cond / 10)

            if iid_dataset:
                factor_sqrt = torch.sqrt(self.snr_base / snr_nat).reshape([-1, 1, 1])
                measurments = factor_sqrt * noise + self.phi(p).unsqueeze(1)
            else:
                factor_sqrt = torch.sqrt(self.snr_base / snr_nat).reshape([-1, 1])
                measurments = factor_sqrt * noise + self.phi(p)
            return pru.NumpyDataset(measurments.cpu().numpy(), p.cpu().numpy(), metadata=cond.cpu().numpy(),
                                    transform=None)


if __name__ == '__main__':
    problem = FrequencyEstimationMultitone(16, 2, 1, 6, 6, noise_type=NoiseType.GaussianScale)
    # noise = problem._samples_noise(64000, True)
    # cov = torch.mean(torch.permute(noise, dims=[0, 2, 1]) @ noise, dim=0)
    # print(problem.prior.mean())

    # def autocorr(x, lags=None):
    #     '''numpy.correlate, non partial'''
    #     mean = x.mean()
    #     var = np.var(x)
    #     xp = x - mean
    #     corr = np.correlate(xp, xp, 'full')[len(x) - 1:] / var / len(x)
    #
    #     return corr if lags is None else corr[:lags]
    #
    #
    # # autocorr_v = np.stack([autocorr(noise.cpu().numpy()[i, 0, :]) for i in range(noise.shape[0])], axis=0).mean(axis=0)
    # cov = torch.mean(torch.permute(noise, dims=[0, 2, 1]) @ noise, dim=0)
    # # cov_real = torch.real(cov)
    # from matplotlib import pyplot as plt
    #
    # plt.matshow(cov.cpu().numpy())
    # plt.colorbar()
    # plt.tight_layout()
    #
    # # plt.savefig("noise_correlation_time_seq.png")
    # # plt.savefig("noise_correlation_time_seq.svg")
    # plt.show()
    #
    # norm = distributions.MultivariateNormal(torch.zeros(16, device=cov.device), covariance_matrix=cov)
    # noise_gauss = norm.sample([64000])
    # # autocorr_gauss = np.stack([autocorr(noise_gauss.cpu().numpy()[i, :]) for i in range(noise_gauss.shape[0])], axis=0)
    # # autocorr_gauss = np.mean(autocorr_gauss, axis=0)
    # fft_gauss = np.mean(np.abs(np.fft.fft(noise_gauss.cpu().numpy(), axis=-1)), axis=0)
    # fft_real = np.mean(np.abs(np.fft.fft(noise.cpu().numpy(), axis=-1)), axis=0).flatten()
    # fft_gauss = np.concatenate([fft_gauss[fft_gauss.shape[-1] // 2:], fft_gauss[:fft_gauss.shape[-1] // 2]])
    # fft_real = np.concatenate([fft_real[fft_real.shape[-1] // 2:], fft_real[:fft_real.shape[-1] // 2]])
    # x = np.linspace(-np.pi, np.pi - 2 * np.pi / 16, 16)
    # plt.plot(x, fft_real.flatten(), label="Real noise")
    # plt.plot(x, fft_gauss.flatten(), label="Gaussian noise")
    # plt.legend()
    # plt.grid()
    # plt.xlabel("Frequency")
    # plt.ylabel("Magnitude")
    # plt.show()
    # print("a")
    # # plt.plot(autocorr_v, label="Real noise")
    # # plt.plot(autocorr_gauss, label="Gaussian noise")
    # # plt.legend()
    # # plt.grid()
    # # plt.show()
    #
    # # dataset = problem.get_dataset(128000, cond=problem.get_condition_list()[0], iid_dataset=True)
    # raise NotImplementedError
    opt_likelihood = problem.get_optimal_likelihood_score()
    opt_prior = problem.get_optimal_prior_score()
    for snr in problem.get_condition_list():
        # print(snr)
        _, fim_ref, prior_fim_ref = problem.bcrb(snr, 1)
        dataset = problem.get_dataset(64000, cond=snr, iid_dataset=True)
        dl = torch.utils.data.DataLoader(dataset, batch_size=512)
        fim = 0
        pfim = 0
        for x, theta, meta in dl:
            x = x.to(problem.device)
            theta = theta.to(problem.device)
            meta = meta.to(problem.device)
            r = opt_likelihood(x, theta, meta)
            fim += torch.mean(r.unsqueeze(dim=-1) @ r.unsqueeze(dim=-2), dim=0)

            p = opt_prior(theta)
            pfim += torch.mean(p.unsqueeze(dim=-1) @ p.unsqueeze(dim=-2), dim=0)

        fim /= len(dl)
        pfim /= len(dl)
        print("-" * 100)
        print(100 * torch.linalg.norm(pfim - prior_fim_ref, ord=2) / torch.linalg.norm(prior_fim_ref, ord=2))
        print(100 * torch.linalg.norm(fim - fim_ref, ord=2) / torch.linalg.norm(fim_ref, ord=2))
        print(snr)
        print(1 / (fim + pfim), fim, pfim)
