import torch
from torch import nn

from utils.config import ScoreModelType



class BaseScoreModel(nn.Module):
    def __init__(self,
                 in_m,
                 in_prior_model,
                 in_likelihood_model,
                 score_model_type,
                 is_complex: bool = False,
                 theta_injection: bool = False,
                 theta_project=None):
        super().__init__()
        self.m = in_m
        self.is_complex = is_complex
        self.add_module("prior_model", in_prior_model)
        self.add_module("likelihood_model", in_likelihood_model)
        if theta_project is not None:
            self.add_module("theta_project", theta_project)
        else:
            self.theta_project = None


        self.score_model_type = score_model_type
        self.optimal_prior = False
        self.optimal_prior_score = None
        self.theta_injection = theta_injection


    def set_optimal_prior(self, optimal_prior):
        self.optimal_prior = True
        self.optimal_prior_score = optimal_prior

    # TODO:Remove this methods
    @property
    def mlp_prior(self):
        return self.prior_model

    @property
    def mlp(self):
        return self.likelihood_model

    def likelihood_score(self, input_x, input_theta, input_meta, sum_over_iid=True):
        """

        :param input_x: [batch_size, n, k]
        :param input_theta: [batch_size, m]
        :param input_meta: [batch_size, m]
        :param sum_over_iid: bool
        :return: [batch_size, n]
        """
        if self.is_complex:
            x_real = torch.real(input_x)
            x_imag = torch.imag(input_x)
            input_x = torch.cat([x_real, x_imag], dim=-1)
        if self.theta_project is not None:
            input_theta = self.theta_project(input_theta)
        if self.score_model_type == ScoreModelType.PriorLikelihoodSplitIID:
            theta = input_theta.unsqueeze(dim=1).repeat([1, input_x.shape[1], 1])
            data = torch.cat([input_x, theta], dim=-1)
            meta = input_meta.unsqueeze(dim=1).repeat([1, input_x.shape[1]])
            if self.theta_injection:
                likelihood_score = self.likelihood_model(data, meta, input_theta)
            else:
                likelihood_score = self.likelihood_model(data, meta)
        elif self.score_model_type == ScoreModelType.PriorLikelihoodSplitIIDModelInformed:
            likelihood_score = self.likelihood_model(input_x, input_meta, input_theta)
        else:
            data = torch.cat([input_x.reshape([input_x.shape[0], -1]), input_theta], dim=-1)
            meta = input_meta.reshape([input_x.shape[0]])
            if self.theta_injection:
                likelihood_score = self.likelihood_model(data, meta, input_theta)
            else:
                likelihood_score = self.likelihood_model(data, meta)
        if sum_over_iid and self.score_model_type in [ScoreModelType.PriorLikelihoodSplitIID,
                                                      ScoreModelType.PriorLikelihoodSplitIIDModelInformed]:
            likelihood_score = torch.sum(likelihood_score, dim=1)
        return likelihood_score

    def prior_score(self, input_theta):
        if self.optimal_prior:
            prior_score = self.optimal_prior_score(input_theta)
        else:
            prior_score = self.prior_model(input_theta)
        return prior_score

    def forward(self, input_x, input_theta, input_meta):
        likelihood_score = self.likelihood_score(input_x, input_theta, input_meta)
        prior_score = self.prior_score(input_theta)
        if self.score_model_type == ScoreModelType.Posterior:
            return likelihood_score + prior_score
        elif self.score_model_type in [ScoreModelType.PriorLikelihoodSplit, ScoreModelType.PriorLikelihoodSplitIID,ScoreModelType.PriorLikelihoodSplitIIDModelInformed]:
            return likelihood_score, prior_score
        else:
            raise Exception("Unknown score model type")

    def get_prior_parameters(self):
        return self.mlp_prior[0].weight

    def get_likelihood_parameters(self):
        return self.mlp.parameters()
