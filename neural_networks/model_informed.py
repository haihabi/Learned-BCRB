import torch

from torch import nn


class ModelInformation(nn.Module):
    def __init__(self, tau_size, is_complex=False):
        super().__init__()
        self.tau_size = tau_size
        self.is_complex = is_complex

    def output_size(self):
        return self.tau_size * (1 + int(self.is_complex))

    def forward(self, p):
        raise NotImplementedError

    def get_jacobian(self, p):
        raise NotImplementedError


class ModelInformationLikelihood(nn.Module):
    def __init__(self, mi: ModelInformation, score_model):
        super().__init__()
        self.add_module("mi", mi)
        self.add_module("score_model", score_model)

    def forward(self, x, meta, p):

        tau = self.mi(p).unsqueeze(dim=1).repeat([1, x.shape[1], 1])
        if self.mi.is_complex:
            tau_real = torch.real(tau)
            tau_imag = torch.imag(tau)
            tau = torch.cat([tau_real, tau_imag], dim=-1)
        data = torch.cat([x, tau], dim=-1)
        meta = meta.unsqueeze(dim=1).repeat([1, x.shape[1]])
        z = self.score_model(data, meta, tau)
        J = self.mi.get_jacobian(p)
        if self.mi.is_complex:
            J_real = torch.real(J)
            J_imag = torch.imag(J)
            J = torch.cat([J_real, J_imag], dim=-2)
        if len(J.shape) == 2:
            y = z @ J
            return y
        else:
            return z @ J
