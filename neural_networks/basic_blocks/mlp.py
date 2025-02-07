import torch
from torch import nn

from neural_networks.basic_blocks.adpative_normalization import SelectiveAdaptiveNormalization
from neural_networks.basic_blocks.se_block import SEBlock


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, feature_size, n_layers, non_linearity=nn.ReLU, normalization=None,
                 bias_output=False, se_block=False, ratio=16, droupout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, feature_size))
        for _ in range(n_layers):
            self.layers.append(nn.Linear(feature_size, feature_size))
            if normalization is not None:
                self.layers.append(normalization())
            self.layers.append(non_linearity())
            if droupout > 0:
                self.layers.append(nn.Dropout(droupout))
            if se_block:
                self.layers.append(SEBlock(feature_size, non_linearity, ratio))
        self.layers.append(nn.Linear(feature_size, out_dim, bias=bias_output))

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:

            if isinstance(layer, SelectiveAdaptiveNormalization):
                x = layer(x, args[0])
            else:
                x = layer(x)
        return x


class LearnedScale(nn.Module):
    def __init__(self, base_module, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.base_module = base_module

    def forward(self, x):
        return self.scale.reshape([*[1 for i in range(len(x.shape) - 1)], -1]) * self.base_module(x)


class Injection(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.linear_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.scale_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), LearnedScale(nn.Tanh(), hidden_dim))

    def forward(self, x, y):
        condition = y
        shift = self.linear_layer(condition)
        scale = self.scale_layer(condition)
        if len(shift.shape) == 2 and len(x.shape)==3:  # if condition is has no i.i.d. dimension add it.
            shift = shift.unsqueeze(dim=1)
            scale = scale.unsqueeze(dim=1)
        return x * scale + shift


class MLPInject(nn.Module):
    def __init__(self, in_dim, out_dim, in_theta_dim, feature_size, n_layers, non_linearity=nn.ReLU, normalization=None,
                 bias_output=True, se_block=False, ratio=16, droupout=0.0, transformer=False, bypass=False,
                 non_linearity_normalization=True,
                 inject=False,output_rescale=False):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers == 1:
            # if inject: self.layers.append(Injection(in_dim, in_theta_dim))
            self.layers.append(nn.Linear(in_dim, out_dim, bias=bias_output))
            torch.nn.init.xavier_normal_(self.layers[-1].weight)
            if normalization is not None:
                self.layers.append(normalization())
        else:
            self.layers.append(nn.Linear(in_dim, feature_size))
            for _ in range(n_layers):

                if bypass:
                    self.layers.append(Injection(feature_size, in_theta_dim))
                    if normalization is not None:
                        self.layers.append(normalization())

                    class ByPass(nn.Module):
                        def __init__(self, in_layers):
                            super().__init__()
                            self.m = nn.Sequential(*in_layers)

                        def forward(self, x):
                            return x + self.m(x)

                    _layers = []
                    _layers.append(nn.Linear(feature_size, feature_size))

                    _layers.append(non_linearity())
                    if se_block:
                        _layers.append(SEBlock(feature_size, non_linearity, ratio))
                    if droupout > 0:
                        _layers.append(nn.Dropout(droupout))

                    self.layers.append(ByPass(_layers))
                else:
                    if inject: self.layers.append(Injection(feature_size, in_theta_dim))
                    self.layers.append(nn.Linear(feature_size, feature_size))

                    if normalization is not None and non_linearity_normalization:
                        self.layers.append(normalization())

                    self.layers.append(non_linearity())

                    if se_block:
                        self.layers.append(SEBlock(feature_size, non_linearity, ratio))
                    if droupout > 0:
                        self.layers.append(nn.Dropout(droupout))

            if inject and output_rescale: self.layers.append(Injection(feature_size, in_theta_dim))
            self.layers.append(nn.Linear(feature_size, out_dim, bias=bias_output))
            if normalization is not None and output_rescale:
                self.layers.append(normalization())

    def forward(self, x, *args, **kwargs):
        for layer in self.layers:

            if isinstance(layer, SelectiveAdaptiveNormalization):
                x = layer(x, args[0])
            elif isinstance(layer, Injection):
                x = layer(x, args[1])
            else:
                x = layer(x)
        return x
