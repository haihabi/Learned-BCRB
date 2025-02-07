import torch
import torch.autograd as autograd
from torch.nn.modules.loss import _Loss


def score_norm(in_score, scale=None):
    if scale is None:
        return 0.5 * (torch.norm(in_score, dim=-1)) ** 2
    else:
        return 0.5 * torch.sum((in_score ** 2) * scale, dim=-1)


def batch_compute_jac(in_score, in_data, scale=None):
    grad_list = []
    for i in range(in_score.shape[1]):
        if scale is None:
            outputs = in_score[:, i]
        else:
            outputs = in_score[:, i] * scale[:, i]
        gradients = autograd.grad(outputs=torch.sum(outputs, dim=0), inputs=in_data,
                                  create_graph=True, retain_graph=True)[0]
        grad_list.append(gradients)
    return torch.stack(grad_list, dim=-1)


class ScoreMatchingLoss(_Loss):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, in_score, in_data):
        jac = batch_compute_jac(in_score, in_data)
        jac_trace = torch.diagonal(jac, dim1=1, dim2=2).sum(dim=-1)
        loss_base = score_norm(in_score) + jac_trace
        if self.reduction == 'mean':
            return torch.mean(loss_base)
        else:
            return loss_base



class ScoreMatchingLossLikelihood(_Loss):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.base_loss = ScoreMatchingLoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, in_likelihood, in_prior, in_data):
        if len(in_likelihood.shape) == 3:
            jac = batch_compute_jac(in_likelihood.mean(1), in_data)
            ss=score_norm(in_likelihood).mean(1)+torch.diagonal(jac, dim1=1, dim2=2).sum(dim=-1)
            lc = torch.sum(in_likelihood.mean(1) * in_prior, dim=-1)
        else:
            ss = self.base_loss(in_likelihood, in_data)
            lc = torch.sum(in_likelihood * in_prior, dim=-1)
        if self.reduction == 'mean':
            return torch.mean(ss + lc)
        else:
            return ss + lc
