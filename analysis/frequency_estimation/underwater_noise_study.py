from matplotlib import pyplot as plt
from signal_models.frequency_estimation.frequency_estimation_utils import load_wav_dataset, compute_covariance_matrix

size = 1200000

data_filter = load_wav_dataset(size, 1, 16)
xxt = compute_covariance_matrix(data_filter)
plt.matshow(xxt)
plt.colorbar()
plt.show()
