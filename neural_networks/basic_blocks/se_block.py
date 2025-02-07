import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, non_linearity, ratio=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            non_linearity(),
            nn.Linear(in_channels // ratio, in_channels),
            nn.Sigmoid()
        )
        self.se[2].bias.data=4*torch.ones_like(self.se[2].bias.data) # Initialized such that SE Block will act as bypass

    def forward(self, x):
        return x * self.se(x)
