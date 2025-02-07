from analysis.helpers import compute_re_vs_param
from matplotlib import pyplot as plt
import pyresearchutils as pru
import numpy as np

if __name__ == '__main__':
    db = pru.signal_processing.db

    gbcrb, _, rho_list, mmse_error, snr_array = compute_re_vs_param(
        ["solar-wildflower-1396", "fragrant-disco-1397"],
        run_est=True,
        number_of_samples=256000,
        param_name="rho_cov")

    x = db(snr_array.cpu().numpy())
    for gcrb, rho,mmse in zip(gbcrb, rho_list,mmse_error):
        res = db(gcrb.flatten())
        print(mmse)
        plt.plot(x, res, label=r"LBCRB @ $\rho$" + f"={rho}")
        plt.plot(x, db(mmse),"--o", label="MMSE @" +r"$\rho$"+ f"={rho}")

    gbcrb, _, rho_list, mmse_error, snr_array = compute_re_vs_param(
        ["solar-wildflower-1396"],
        run_est=False,
        compute_optimal=True,
        number_of_samples=256000,
        param_name="rho_cov")
    for gcrb, rho in zip(gbcrb, rho_list):
        res = db(gcrb.flatten())
        plt.plot(x, res,linestyle=(0, (5, 10)),marker="D", label=r"LBCRB (True Score) @ $\rho$" + f"={rho}")
    plt.axvspan(8, np.max(x), color='red', alpha=0.5, label="No Information Region")
    plt.axvspan(np.min(x), -7, color='red', alpha=0.5)
    plt.xlabel('SNR [dB]')
    plt.ylabel('MSE [dB]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("qlinear_correlated_noise.svg")
    plt.show()

    gbcrb, _, rho_list, _, snr_array = compute_re_vs_param(
        ["solar-wildflower-1396", "fallen-tree-1405", "crimson-salad-1404"
            , "colorful-cherry-1403", "hearty-breeze-1402", "smooth-firefly-1401", "drawn-dust-1400",
         "olive-puddle-1399",
         "drawn-elevator-1398", "fragrant-disco-1397"],
        number_of_samples=256000,
        param_name="rho_cov")

    results_matrix = []
    for gcrb, rho in zip(gbcrb, rho_list):
        res = db(gcrb.flatten())
        plt.plot(x, res, label=r"$\rho$" + f"={rho}")
        results_matrix.append(res)

    plt.axvspan(8, np.max(x), color='red', alpha=0.5, label="No Information Region")
    plt.axvspan(np.min(x), -7, color='red', alpha=0.5)
    plt.xlabel('SNR [dB]')
    plt.ylabel('MSE [dB]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("score_bcrb_vs_snr_over_rho_values.svg")
    plt.show()
    results_matrix = np.asarray(results_matrix)
    print("a")
    plt.plot(rho_list, results_matrix[:, 11])
    plt.ylabel('MSE [dB]')
    plt.xlabel(r"$\rho$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("score_bcrb_vs_rho.svg")
    plt.show()
