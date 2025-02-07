import torch
import pyresearchutils as pru
from torch import nn


class BasePrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = pru.get_working_device()
        self.one = torch.ones(1, device=self.device)

    def sample(self, n_samples):
        raise NotImplementedError

    def prior_fim(self):
        raise NotImplementedError

    def prior_score(self, p):
        raise NotImplementedError
