from torch import nn

from neural_networks.basic_blocks.selective_weight import Selective


class SelectiveAdaptiveNormalization(nn.Module):
    def __init__(self, input_dim, in_meta_options, eps=1e-6):
        super().__init__()
        self.input_dim = input_dim
        self.sel = Selective(in_meta_options)
        self.eps = eps

    def forward(self, x, y):
        return x * self.sel(y).unsqueeze(-1)
