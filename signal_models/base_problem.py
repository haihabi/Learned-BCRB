from enum import Enum
from typing import Callable, List

from torch import nn
import torch
import pyresearchutils as pru
from neural_networks.model_informed import ModelInformation


class BaseProblem(nn.Module):

    def __init__(self, n, m, k, condition_list, has_bcrb=False, is_complex=False, has_score=False):
        super().__init__()
        """
        Initialize the BaseProblem class.
        :param n: The number of measurements.
        :param m: The number of parameters.
        :param k: The number of i.i.d. samples.
        :param has_bcrb: A boolean. True if the problem has the Bayesian Cramer Rao Bound.
        :param is_complex: A boolean. True if the measurement has complex values.
        """
        self.n = n
        self.m = m
        self.k = k
        self.has_bcrb = has_bcrb
        self.has_score = has_score
        self.is_complex = is_complex
        self.device = pru.get_working_device()
        self.one = torch.ones(1, device=self.device)
        self.condition_list = condition_list.to(self.device)

    def phi(self, p):
        """
        Generate a signal from the given parameters.
        :param p: The parameter theta.
        :return: The signal.
        """
        raise NotImplementedError

    def dphidtheta(self, p):
        """
        Compute the derivative of the signal with respect to the parameters.
        :param p: The parameter theta.
        :return: The derivative of the signal with respect to the parameters.
        """
        raise NotImplementedError

    def get_condition_list(self) -> List:
        return self.condition_list

    def get_model_information(self) -> ModelInformation:
        base = self

        class ModelInformationLinear(ModelInformation):
            def __init__(self):
                super().__init__(tau_size=base.n, is_complex=base.is_complex)

            def forward(self, p):
                return base.phi(p)

            def get_jacobian(self, p):
                return base.dphidtheta(p).reshape([-1, base.n, base.m])

        return ModelInformationLinear()

    def save_model(self, path):
        """
        Save the model to the given path.
        :param path: A string. The path to save the model.
        :return: None
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        Load the model from the given path.
        :param path: A string. The path to load the model.
        :return: None
        """
        self.load_state_dict(torch.load(path))

    def get_prior_fim(self):
        """
        Get the prior Fisher Information Matrix.
        :return: The prior Fisher Information Matrix.
        """
        raise NotImplementedError

    def get_expected_likelihood_fim(self, snr, k_iid):
        """
        Get the expected Fisher Information Matrix.
        :param snr: The signal to noise ratio.
        :param k_iid: The number of iid samples.
        :return: The expected Fisher Information Matrix.
        """
        raise NotImplementedError

    def bcrb(self, snr: float, k_iid: int = None):
        """
        Get the Bayesian Cramer Rao Bound.
        :param snr: The signal to noise ratio.
        :param k_iid: The number of iid samples.
        :return: The Bayesian Cramer Rao Bound.
        """
        if k_iid is None:
            k_iid = self.k
        efim = self.get_expected_likelihood_fim(snr, k_iid)
        pfim = self.get_prior_fim()
        return torch.linalg.inv(efim + pfim), efim, pfim

    def bcrb_vs_condition(self):
        """
        Get the Bayesian Cramer Rao Bound for all conditions.
        :return: The Bayesian Cramer Rao Bound for all conditions.
        """
        bcrb_list = []
        for condition in self.condition_list:
            bcrb_list.append(torch.stack(self.bcrb(condition)))
        return torch.stack(bcrb_list)

    def get_optimal_posterior_score(self) -> Callable:
        """
        Get the optimal score function.
        :return: The optimal score function.  The function signature is as follows:
        def optimal_score_function(in_x, in_theta, in_sigma):
            return score

        """
        opt_lik = self.get_optimal_likelihood_score()
        opt_prior = self.get_optimal_prior_score()

        def optimal_score_function(in_x, in_theta, in_sigma):
            return opt_lik(in_x, in_theta, in_sigma) + opt_prior(in_theta)

        return optimal_score_function

    def get_optimal_likelihood_score(self) -> Callable:
        raise NotImplementedError

    def get_optimal_prior_score(self) -> Callable:
        raise NotImplementedError


class PriorType(Enum):
    GAUSSIAN = 0
    BETA = 1


def compute_problem_fims(in_problem, in_snr, n_samples=64000, batch_size=512):
    opt_likelihood = in_problem.get_optimal_likelihood_score()
    opt_prior = in_problem.get_optimal_prior_score()
    dataset = in_problem.get_dataset(n_samples, cond=in_snr, iid_dataset=True)
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    fim = 0
    pfim = 0
    for x, theta, meta in dl:
        x = x.to(in_problem.device)
        theta = theta.to(in_problem.device)
        meta = meta.to(in_problem.device)
        r = opt_likelihood(x, theta, meta)
        fim += torch.mean(r.unsqueeze(dim=-1) @ r.unsqueeze(dim=-2), dim=0)

        p = opt_prior(theta)
        pfim += torch.mean(p.unsqueeze(dim=-1) @ p.unsqueeze(dim=-2), dim=0)
    fim /= len(dl)
    pfim /= len(dl)
    return fim, pfim
