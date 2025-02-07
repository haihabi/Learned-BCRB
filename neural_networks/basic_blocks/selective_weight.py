import torch
from torch import nn


class Selective(nn.Module):
    def __init__(self, option_list, init_weights=None, trainable=True):
        super().__init__()
        self.n = len(option_list)
        self.weights = nn.Parameter(torch.ones(self.n) if init_weights is None else init_weights,
                                    requires_grad=trainable)
        self.option_list = torch.tensor(option_list, requires_grad=False) if not isinstance(option_list,
                                                                                            torch.Tensor) else option_list.clone().detach()

    def forward(self, option):
        i = torch.where(self.option_list.reshape([1, -1]).to(option.device) == option.reshape([-1, 1]))[-1]
        return self.weights[i].reshape(option.shape)

    def update_weights(self, new_weights):
        self.weights.data = new_weights
