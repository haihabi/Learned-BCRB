import torch
from enum import Enum
from torch import nn



class NonLinearityType(Enum):
    SILU = 0
    RELU = 1
    TANH = 2
    GELU = 3
    SIGMOID = 4
    ELU = 5
    CELU = 6
    MISH = 7
    GCU = 8


class GCU(nn.Module):
    def forward(self, x):
        return x * torch.cos(x)


NON_LINEARITY_TYPE_DICT = {NonLinearityType.SILU: nn.SiLU, NonLinearityType.RELU: nn.ReLU,
                           NonLinearityType.TANH: nn.Tanh,
                           NonLinearityType.GCU: GCU,
                           NonLinearityType.GELU: nn.GELU, NonLinearityType.SIGMOID: nn.Sigmoid,
                           NonLinearityType.ELU: nn.ELU, NonLinearityType.CELU: nn.CELU, NonLinearityType.MISH: nn.Mish}

NON_LINEARITY_LIST = [k.name for k in list(NON_LINEARITY_TYPE_DICT.keys())]


def get_non_linearity(in_non_linearity_type: NonLinearityType):
    return NON_LINEARITY_TYPE_DICT[in_non_linearity_type]
