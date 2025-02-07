from signal_models import LinearGaussianProblem
import pyresearchutils as pru
from signal_models.frequency_estimation.frequency_estimation_multitone import FrequencyEstimationMultitone
from signal_models.quantization.quantized_linear_gaussian import QuantizedLinearProblem
from utils.config import SignalModelType

SIGNAL_MODEL_TYPE_DICT = {SignalModelType.LinearGamma: LinearGaussianProblem,
                          SignalModelType.QuantizedLinear: QuantizedLinearProblem,
                          SignalModelType.MultitoneFreq: FrequencyEstimationMultitone}


def get_signal_model(in_signal_model_type: SignalModelType, *args, **kwargs):
    return SIGNAL_MODEL_TYPE_DICT[in_signal_model_type](*args, **kwargs).to(pru.get_working_device())
