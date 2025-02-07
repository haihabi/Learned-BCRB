import os

import numpy as np
import scipy
from tqdm import tqdm


def psd2signal(psd):
    psd_ratio = 10 ** (psd / 10)
    magnitude = np.sqrt(psd_ratio)
    magnitude = np.concatenate([np.flip(magnitude[1:, :]), magnitude], axis=0)
    phase = np.pi * np.random.rand(*magnitude.shape)  # random phase
    complex = magnitude * np.exp(1j * phase)
    signal = np.fft.ifft(complex, axis=0)
    signal = signal / np.sqrt(np.var(signal))
    return signal


def get_alpha_beta_values(in_m, in_v, a_h, a_l):
    """
    Get alpha and beta values for the beta distribution given the mean and variance.
    :param in_m:  Mean of the distribution
    :param in_v:  Variance of the distribution
    :param a_h:  Upper bound of the distribution
    :param a_l:  Lower bound of the distribution
    :return:
    """
    delta_a = a_h - a_l
    delta_alpha = (delta_a / (in_m - a_l) - 1)
    _alpha = (delta_alpha * delta_a ** 2 - in_v * (1 + delta_alpha) ** 2) / (in_v * (1 + delta_alpha) ** 3)
    if _alpha < 0:
        raise Exception("aa")
    _beta = _alpha * delta_alpha
    if _alpha < 2:
        raise Exception("")

    if _beta < 2:
        raise Exception("")

    return _alpha, _beta


def find_wav_files(folder):
    """
    Finds all WAV files in a given folder and its subfolders.

    Args:
      folder: The path to the folder to search.

    Returns:
      A list of WAV files in the folder and its subfolders.
    """
    wav_files = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, filename))
    return wav_files


def load_wav_dataset(size, k, n, folder_path="/data/datasets/QiandaoEar22"):
    """
    Load a dataset from a folder containing WAV files.
    :param k: Number of i.i.d. samples.
    :param n: Number of samples in each i.i.d. sample.
    :param folder_path: The path to the folder containing the WAV files.
    :return:
    """
    files = find_wav_files(folder_path)
    res_list = []
    for file in files:
        d0, d1 = scipy.io.wavfile.read(file)
        norm_d = d1 / np.std(d1)
        d_seq = norm_d.reshape([-1, k, n])
        res_list.append(d_seq)
    data_full = np.concatenate(res_list)
    del res_list
    np.random.shuffle(data_full)
    data_filter = data_full[:size, :, :]
    del data_full
    return data_filter


def compute_covariance_matrix(in_data_filter):
    """
    Compute the covariance matrix of a given dataset.
    :param in_data_filter:
    :return:
    """
    mu = np.mean(in_data_filter, axis=0, keepdims=True)
    mu_remove = in_data_filter - mu
    return np.mean(np.transpose(mu_remove, (0, 2, 1)) @ mu_remove, axis=0)
