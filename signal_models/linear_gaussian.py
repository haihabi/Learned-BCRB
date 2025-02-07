import torch
import pyresearchutils as pru
from torch import distributions
from typing import Callable, Mapping, Any

from neural_networks.model_informed import ModelInformation
from signal_models.base_problem import BaseProblem, compute_problem_fims
from torch import nn


class LinearGaussianProblem(BaseProblem):
    def __init__(self, n, m, k, alpha_rho, beta_rho, minimal_snr=-2, maximal_snr=2, snr_points=20,
                 *args, **kwargs):
        """
        This class implements the linear model with a Gaussian prior.
        :param n:  Number of measurements (int)
        :param m:  Number of parameters (int)
        :param k:  Number of i.i.d. samples (int)
        :param alpha_rho:  Shape parameter of the Gamma distribution (float)
        :param beta_rho: Rate parameter of the Gamma distribution (float)
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        """
        condition_list = 10 * torch.log10(torch.logspace(minimal_snr, maximal_snr, snr_points)).to(
            pru.get_working_device())
        super().__init__(n, m, k, condition_list, has_bcrb=True)

        self.alpha_rho = alpha_rho
        self.beta_rho = beta_rho
        self.device = pru.get_working_device()
        # Generate Mixing matrix
        a_matrix = torch.randn([n, m]).to(self.device)  # Mixing matrix
        a_matrix /= torch.linalg.norm(a_matrix, ord=2)  # Normalize the mixing matrix to have a norm of 1
        self.register_buffer("a_matrix", a_matrix)

        # Generate Covariance matrix for the noise
        b_matrix = torch.randn([n, n]).to(self.device)
        cov = b_matrix @ b_matrix.T
        cov /= torch.trace(cov)  # Normalize the covariance matrix to have a trace of 1

        self.register_buffer("cov_inv", torch.linalg.inv(cov))
        self.register_buffer("cov", cov)
        self.recompute_snr_base()

    def get_model_information(self) -> ModelInformation:
        base = self

        class ModelInformationLinear(ModelInformation):
            def __init__(self):
                super().__init__(tau_size=base.n)
                self.a_matrix = nn.Parameter(base.a_matrix.clone().detach(), requires_grad=False)

            def forward(self, p):
                return p @ self.a_matrix.T

            def get_jacobian(self, p):
                return self.a_matrix

        return ModelInformationLinear()

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        super().load_state_dict(state_dict, strict)
        self.recompute_snr_base()

    def recompute_snr_base(self):
        pow_theta = self.beta_rho ** 2 * torch.eye(self.m).to(self.device)

        _snr = torch.trace(self.a_matrix @ pow_theta @ self.a_matrix.T) / torch.trace(self.cov)
        self.snr_base = _snr.item()

    def get_prior_fim(self):
        """
        Get the prior Fisher Information Matrix.
        :return: a matrix of the prior Fisher Information Matrix.
        """
        return (1 / self.beta_rho ** 2) * torch.eye(self.m).to(self.device)

    def get_expected_likelihood_fim(self, snr, k_iid):
        snr_nat = 10 ** (snr / 10)
        cov_snr_factor = self.snr_base / snr_nat
        fim = k_iid * self.a_matrix.T @ torch.linalg.inv(cov_snr_factor * self.cov) @ self.a_matrix
        return fim


    def get_optimal_likelihood_score(self) -> Callable:
        def optimal_score_function(in_x, in_theta, in_snr=None,split_iid=False):
            snr_nat = 10 ** (in_snr / 10)
            one_over_factor = (snr_nat / self.snr_base).reshape([-1, 1, 1])
            delta = one_over_factor * (in_x - (self.a_matrix @ in_theta.T).T.unsqueeze(dim=1)) @ (
                    self.a_matrix.T @ self.cov_inv).T
            if split_iid:
                return delta
            return torch.sum(delta, dim=1)

        return optimal_score_function

    def get_optimal_prior_score(self) -> Callable:
        def optimal_score_function(in_theta):
            # prior_score = ((self.alpha_rho - 1) / in_theta) - self.beta_rho
            return -in_theta / self.beta_rho ** 2
            # return prior_score

        return optimal_score_function

    def get_dataset(self, dataset_size, iid_dataset=True, cond=None):
        noise = distributions.MultivariateNormal(torch.zeros(self.n).to(self.device), covariance_matrix=self.cov)
        parameter = distributions.MultivariateNormal(torch.zeros(self.m).to(self.device),
                                                     covariance_matrix=torch.eye(self.m).to(
                                                         self.device) * self.beta_rho ** 2)
        snr_list = self.get_condition_list()

        p = parameter.sample([dataset_size]).to(self.device)
        if iid_dataset:
            noise = noise.sample([dataset_size, self.k]).to(self.device)
        else:
            noise = noise.sample([dataset_size]).to(self.device)

        if cond is None:
            index = torch.randint(low=0, high=len(snr_list), size=(dataset_size, 1)).flatten()
            cond = torch.tensor(snr_list)[index].to(self.device)
        else:
            cond = torch.ones(dataset_size).to(self.device) * cond
        snr_nat = 10 ** (cond / 10)

        if iid_dataset:
            factor_sqrt = torch.sqrt(self.snr_base / snr_nat).reshape([-1, 1, 1])
            measurments = factor_sqrt * noise + (self.a_matrix @ p.T).T.unsqueeze(1)
        else:
            factor_sqrt = torch.sqrt(self.snr_base / snr_nat).reshape([-1, 1])
            measurments = factor_sqrt * noise + (self.a_matrix @ p.T).T
        return pru.NumpyDataset(measurments.cpu().numpy(), p.cpu().numpy(), metadata=cond.cpu().numpy(),
                                transform=None)


if __name__ == '__main__':
    problem = LinearGaussianProblem(10, 4, 1, 1, 2)

    for snr in problem.get_condition_list():
        fim, pfim = compute_problem_fims(problem, snr)
        _, fim_ref, prior_fim_ref = problem.bcrb(snr, 1)
        print(100 * torch.linalg.norm(pfim - prior_fim_ref, ord=2) / torch.linalg.norm(prior_fim_ref, ord=2))
        print(100 * torch.linalg.norm(fim - fim_ref, ord=2) / torch.linalg.norm(fim_ref, ord=2))
