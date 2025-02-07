from enum import Enum
from typing import List

import torch
from torch import nn
import pyresearchutils as pru
from neural_networks.base_score_model import BaseScoreModel
from utils.config import ScoreLossType, ScoreModelType
from score import score_matching_loss


class TrainingPart(Enum):
    Posterior = 0
    Likelihood = 1
    Prior = 2


def prepare_input(in_param):
    param = in_param.clone().detach()
    return param.requires_grad_(True)


class ScoreLossContext(nn.Module):
    def __init__(self,
                 min_value,
                 max_value,
                 loss_type: ScoreLossType,
                 score_model_type: ScoreModelType = ScoreModelType.Posterior,
                 training_part: TrainingPart = TrainingPart.Prior,
                 in_condition_list: List[float] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.min_value = min_value
        self.max_value = max_value
        self.loss_type = loss_type
        self.device = pru.get_working_device()
        self.base_score_loss = score_matching_loss.ScoreMatchingLoss(reduction="none")
        self.score_loss_lik = score_matching_loss.ScoreMatchingLossLikelihood(reduction="none")
        self.score_model_type = score_model_type
        self.training_part = training_part

    def loss_function(self, in_score_model: BaseScoreModel, in_x, in_param, in_cond, add_str="", in_ema=None):
        final_str = "" if self.training else "_val" + add_str
        if self.score_model_type in [ScoreModelType.PriorLikelihoodSplit,
                                     ScoreModelType.PriorLikelihoodSplitIID,
                                     ScoreModelType.PriorLikelihoodSplitIIDModelInformed]:
            norm = 1 if self.score_model_type == ScoreModelType.PriorLikelihoodSplit else in_x.shape[1]
            norm = 1
            if self.training_part == TrainingPart.Prior:
                raise NotImplementedError
            elif self.training_part == TrainingPart.Likelihood:
                loss_prior_sm = 0
                prior_score = in_score_model.prior_score(in_param).detach()
                outersum_over_iid = True
                if outersum_over_iid:
                    likelihood_score = in_score_model.likelihood_score(in_x, in_param, in_cond, sum_over_iid=False)
                    loss_likelihood_sm_per_sample = self.score_loss_lik(likelihood_score, prior_score,
                                                                        in_param)
                    likelihood_score = torch.sum(likelihood_score, dim=1)
                    # loss_likelihood_sm_per_sample=torch.mean(loss_likelihood_sm_per_sample, dim=1)
                else:
                    likelihood_score = in_score_model.likelihood_score(in_x, in_param, in_cond, sum_over_iid=True)
                    loss_likelihood_sm_per_sample = self.score_loss_lik(likelihood_score, prior_score,
                                                                        in_param)

                loss_likelihood_sm = torch.mean(
                    loss_likelihood_sm_per_sample / norm)  # Divide by the number of iid samples
                loss = loss_likelihood_sm
            else:
                raise ValueError("Incorrect training part.")

            loss_dict = {"loss" + final_str: loss.item(),
                         "loss_prior_sm" + final_str: loss_prior_sm,
                         "loss_likelihood_sm" + final_str: loss_likelihood_sm.item()}
            data_dict = {"likelihood_score": likelihood_score,
                         "loss_likelihood_sm_per_sample": loss_likelihood_sm_per_sample,
                         "prior_score": prior_score,
                         "posterior_score": None}

        elif self.score_model_type == ScoreModelType.Posterior:
            in_score = in_score_model(in_x, in_param, in_cond)
            loss_sm_per_sample = self.base_score_loss(in_score, in_param)

            loss = torch.mean(loss_sm_per_sample)
            loss_dict = {"loss" + final_str: loss.item(),
                         "loss_sm" + final_str: loss.item()}
            data_dict = {"likelihood_score": None,
                         "prior_score": None,
                         "loss_posterior_sm_per_sample": loss_sm_per_sample,
                         "posterior_score": in_score}
        else:
            raise Exception("Unknown score model type")

        return loss, loss_dict, data_dict
