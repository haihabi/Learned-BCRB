from typing import List

import torch
from signal_models.priors.base_prior import BasePrior


class ConcatPrior(BasePrior):
    def __init__(self, list_of_priors: List[BasePrior]):
        super().__init__()
        self.prior_list=list_of_priors
    
    def sample(self, n_samples):
        return torch.cat([p.sample(n_samples) for p in self.prior_list],dim=-1)

    def prior_fim(self):
        fim_list = []
        for prior in self.prior_list:
            fim_list.append(prior.prior_fim())
        concatenated_fim = torch.block_diag(*fim_list)
        return concatenated_fim

    def prior_score(self, p):
        return torch.cat([prior.prior_score(p[:,i]) for i,prior in enumerate(self.prior_list)],dim=-1)

