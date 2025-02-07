import pyresearchutils as pru
from enum import Enum

from neural_networks.basic_blocks.non_linearity_factory import NonLinearityType


class ScoreLossType(Enum):
    SM = 0
    TruncatedSM = 1


class ScoreModelType(Enum):
    Posterior = 0
    PriorLikelihoodSplit = 1
    PriorLikelihoodSplitIID = 2  # Likelihood is computed for each i.i.d. sample
    PriorLikelihoodSplitIIDModelInformed = 3  # Likelihood is computed for each i.i.d. sample

    def is_model_informed(self):
        return self == ScoreModelType.PriorLikelihoodSplitIIDModelInformed

    def is_iid(self):
        return self == ScoreModelType.PriorLikelihoodSplitIID or self == ScoreModelType.PriorLikelihoodSplitIIDModelInformed

    def is_measurement_prior(self):
        return self == ScoreModelType.PriorLikelihoodSplit or self == ScoreModelType.PriorLikelihoodSplitIID or self == ScoreModelType.PriorLikelihoodSplitIIDModelInformed


class NoiseType(Enum):
    GaussianScale = 0
    GaussianDiag = 1
    GaussianMatrix = 2
    Real = 3


def get_default_config():
    cr = pru.initialized_config_reader()
    cr.add_parameter(name="n", default=32, type=int)  # Number of measurements
    cr.add_parameter(name="m", default=1, type=int)  # Number of parameters
    cr.add_parameter(name="k", default=16, type=int)  # Number of i.i.d. samples
    cr.add_parameter(name="snr", default=10.0, type=float)
    cr.add_parameter(name="k_snr", default=6, type=int)
    cr.add_parameter(name="alpha_rho", default=100, type=float)
    cr.add_parameter(name="beta_rho", default=100, type=float)
    cr.add_parameter(name="rho_cov", default=0.9, type=float)
    cr.add_parameter(name="sigma_prior", default=0.25, type=float)
    cr.add_parameter(name="spacing", default=0.25, type=float)
    cr.add_parameter(name="n_bits", default=1, type=int)
    cr.add_parameter(name="n_snrs", default=20, type=int)
    cr.add_parameter(name="minimal_snr", default=-2, type=int)
    cr.add_parameter(name="maximal_snr", default=2, type=int)
    cr.add_parameter(name="threshold", default=1, type=float)
    cr.add_parameter(name="is_random_phase", default=0, type=bool)


    cr.add_parameter(name="noise_type", default="GaussianScale", type=str, enum=NoiseType)
    cr.add_parameter(name="noise_data_folder", default="/data/logs/", type=str)

    cr.add_parameter(name="loss_type", default="SM", type=str, enum=ScoreLossType)
    cr.add_parameter(name="score_model_type", default="PriorLikelihoodSplitIID", type=str,
                     enum=ScoreModelType)
    cr.add_parameter(name="signal_model", default="QuantizedLinear", type=str, enum=SignalModelType)
    cr.add_parameter(name="base_log_folder", default="/data/logs/", type=str)
    cr.add_parameter(name="comment", default="", type=str)

    cr.add_parameter(name="dataset_size", default=1200000, type=int)
    cr.add_parameter(name="dataset_size_eval", default=60000, type=int)
    cr.add_parameter(name="batch_size", default=512, type=int)
    cr.add_parameter(name="warmup_epochs", default=10)
    cr.add_parameter(name="weight_decay", default=0.0, type=float)
    cr.add_parameter(name="weight_decay_prior", default=0.0, type=float)
    cr.add_parameter(name="n_epochs", default=200, type=int)
    cr.add_parameter(name="n_epochs_prior", default=50, type=int)
    cr.add_parameter(name="n_eval", default=200, type=int)
    cr.add_parameter(name="lr", type=float, default=2e-4)
    cr.add_parameter(name="beta1", type=float, default=0.001)
    cr.add_parameter(name="beta2", type=float, default=0.999)
    cr.add_parameter(name="lr_prior", type=float, default=2e-4)
    cr.add_parameter(name="ema_decay", type=float, default=0.9)
    cr.add_parameter(name="div_factor", default=100.0)
    cr.add_parameter(name="optimal_prior", default=0, type=bool)
    cr.add_parameter(name="amsgrad", default=0, type=bool)
    cr.add_parameter(name="optimizer_type", default="AdamW", type=str, enum=OptimizerType)
    cr.add_parameter(name="x_normalization", default=0, type=bool)

    cr.add_parameter(name="n_layers_prior", default=3)
    cr.add_parameter(name="feature_size_prior", default=32)
    cr.add_parameter(name="non_linearity_prior", default="MISH", type=str, enum=NonLinearityType)
    cr.add_parameter(name="se_block_prior", default=0, type=bool)
    cr.add_parameter(name="non_linearity_normalization", default=1, type=bool)
    cr.add_parameter(name="output_bias_prior", default=1, type=bool)
    cr.add_parameter(name="output_rescale", default=1, type=bool)
    cr.add_parameter(name="run_as_posterior", default=0, type=bool)

    cr.add_parameter(name="n_layers", default=3,type=int)
    cr.add_parameter(name="feature_size", default=96, type=int)
    cr.add_parameter(name="non_linearity", default="SIGMOID", type=str, enum=NonLinearityType)
    cr.add_parameter(name="se_block", default=0, type=bool)
    cr.add_parameter(name="inject", default=1, type=bool)
    cr.add_parameter(name="output_bias", default=0, type=bool)

    return cr


class OptimizerType(Enum):
    AdamW = 1
    RMSprop = 2


class SignalModelType(Enum):
    LinearGamma = 0
    QuantizedLinear = 1
    DOA2D = 2
    DOA1D = 3
    Freq = 4
    FreqComplex = 5
    MultitoneFreq = 6
