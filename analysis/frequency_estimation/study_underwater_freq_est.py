from analysis.helpers import load_wandb_run, generate_bound_function, lbcrb_bound
from matplotlib import pyplot as plt
import numpy as np
import pyresearchutils as pru
from analysis.neural_mmse import train_mmse_estimator
import torch
from tqdm import tqdm

if __name__ == '__main__':
    number_of_samples = 64000

    run_name = "sweet-star-83"
    model, problem, param, _ = load_wandb_run(run_name)

    compute_bcrb_real, sigma_array = generate_bound_function(problem, model, number_of_samples=number_of_samples,
                                                             compute_optimal=False)

    problem.change_real2gaussian()
    print("Real to Gaussian")
    print(problem.cov)
    result_function = lbcrb_bound(problem,
                                  split_iid_in_eval=False,
                                  number_of_samples=64000,
                                  batch_size=1024,
                                  compute_optimal=True)

    run_name = "hearty-forest-13"
    model, problem, param, _ = load_wandb_run(run_name)
    compute_bcrb_gaussian, sigma_array = generate_bound_function(problem, model, number_of_samples=number_of_samples,
                                                                 compute_optimal=True)

    res = []
    # res_amp = []
    # res_phase = []

    for sigma in sigma_array:
        _gbcrb_real = compute_bcrb_real(1, sigma)

        # _gbcrb_real_single = compute_bcrb_real_single(1, sigma)
        _gbcrb_gassuain = compute_bcrb_gaussian(1, sigma, opt=True)
        _bcrb = problem.bcrb(sigma)  # Use optimal BCRB
        # _gbcrb_real = _bcrb[0].cpu().detach().numpy()
        res.append(
            np.stack([_gbcrb_real, _gbcrb_gassuain, _bcrb[0].cpu().detach().numpy()], axis=0))
        print(f"BCRB: {sigma} dB")
        print(f"BCRB Real: {_gbcrb_real}")
    res = np.stack(res, axis=0)

    plt.rcParams.update({'font.size': 12})
    # for i in range(res.shape[-1]):
    #
    plt.plot(sigma_array, np.trace(res[:, 0], axis1=1, axis2=2), label="LBCRB Underwater Noise")
    # plt.plot(list(sigma_array)[filer:], images[:, 1], label="NMMSE Underwater Noise")
    a = result_function["optimal"](1)
    print(a)
    plt.plot(sigma_array, np.trace(a, axis1=1, axis2=2), "-x", label="LBCRB Gaussian (UWN Covariance)")
    plt.plot(sigma_array, np.trace(res[:, 1], axis1=1, axis2=2), label="LBCRB WGN")
    plt.plot(sigma_array, np.trace(res[:, 2], axis1=1, axis2=2), "--", label="BCRB WGN")
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xlabel("SNR [dB]")
    plt.ylabel(r"$Tr(\theta)$")
    plt.savefig("frequency_error.svg")
    plt.show()

    # plt.plot(sigma_array, res[:, 0,0,1], label="LBCRB Underwater Noise")
    # # plt.plot(sigma_array, res[:, 1,0,1], label="LBCRB Gaussian")
    # plt.plot(sigma_array, res[:, 2,0,1], "--", label="BCRB")
    # plt.legend()
    # plt.grid()
    # plt.yscale("log")
    # plt.xlabel("SNR [dB]")
    # plt.ylabel(r"$Tr(\theta)$")
    # plt.savefig("frequency_cross.svg")
    # plt.show()

    # plt.plot(sigma_array, np.sqrt(res_amp[:, 0]), label="LBCRB Underwater Noise")
    # plt.plot(sigma_array, np.sqrt(res_amp[:, 1]), label="LBCRB Gaussian")
    # plt.plot(sigma_array, np.sqrt(res_amp[:, 2]),"--", label="BCRB")
    # plt.legend()
    # plt.grid()
    # plt.yscale("log")
    # plt.xlabel("SNR [dB]")
    # plt.ylabel("Frequency Error [Hz]")
    # plt.savefig("frequency_error_2.svg")
    # plt.show()

    # plt.plot(sigma_array, np.sqrt(res_amp[:, 0]), label="LBCRB Underwater Noise")
    # plt.plot(sigma_array, np.sqrt(res_amp[:, 1]), label="LBCRB Gaussian")
    # plt.plot(sigma_array, np.sqrt(res_amp[:, 2]),"--", label="BCRB")
    # plt.legend()
    # plt.grid()
    # plt.yscale("log")
    # plt.xlabel("SNR [dB]")
    # plt.ylabel("Amplitude Error")
    # plt.savefig("amplitude_error.svg")
    # plt.show()
    #
    # plt.plot(sigma_array, np.sqrt(res_phase[:, 0]), label="LBCRB Underwater Noise")
    # plt.plot(sigma_array, np.sqrt(res_phase[:, 1]), label="LBCRB Gaussian")
    # plt.plot(sigma_array, np.sqrt(res_phase[:, 2]),"--", label="BCRB")
    # plt.legend()
    # plt.grid()
    # plt.yscale("log")
    # plt.xlabel("SNR [dB]")
    # plt.ylabel("Phase Error [rad]")
    # plt.savefig("phase_error.svg")
    # plt.show()
