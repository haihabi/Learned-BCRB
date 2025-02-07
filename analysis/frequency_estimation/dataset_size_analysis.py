import matplotlib.pyplot as plt
from analysis.linear_model.study_m_iid import run_analysis_k_study
from analysis.helpers import bcrb_relative_error
import numpy as np

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 12})
    k_iid = 1
    res_dict = {
        1200000: ("sweet-wind-1233", "fluent-dew-1234"),
        120000: ("smart-sound-1241", "desert-surf-1236"),
        12000: ("fresh-blaze-1247", "dark-fog-1238"),
    }

    res_dict = {
        1200000: ("rich-microwave-1280", "colorful-silence-1281"),
        120000: ("celestial-hill-1298", "driven-river-1283"),
        12000: ("bumbling-cosmos-1301", "eager-lion-1285"),
    }

    res_dict = {
        1200000: ("playful-vortex-1315", "electric-grass-1310"),
        120000: ("winter-mountain-1321", "fine-forest-1317"),
        12000: ("elated-puddle-1322", "dulcet-cherry-1319"),
    }

    res_dict = {
        1200000: ("pleasant-sponge-1602", "earnest-puddle-1603"),
        120000: ("rare-sea-1604", "elated-elevator-1605"),
        12000: ("lilac-eon-1606", "peach-bush-1607"),
    }

    res_dict = {
        1200000: ("glad-plasma-1614", "earnest-puddle-1603"),
        120000: ("effortless-wildflower-1615", "elated-elevator-1605"),
        12000: ("lyric-plasma-1616", "peach-bush-1607"),
    }
    res_per_size = []
    for k, (n_non_inf, n_inf) in res_dict.items():
        # k = 16
        n_s = int(k / 20)
        _, _, iid_split, cond, _, _ = run_analysis_k_study(n_non_inf, compute_optimal=True,
                                                           number_of_samples=128000, k=k_iid)  # Only learning error
        _, _, model_base, _, _, _ = run_analysis_k_study(n_inf, compute_optimal=True,
                                                         number_of_samples=128000, k=k_iid)  # Only sampling error
        res_per_size.append([k, np.mean(iid_split), np.mean(model_base)])
        plt.rcParams.update({'font.size': 12})
        plt.plot(cond.cpu().detach().numpy(), iid_split, "--x", label="w/o Physics Encoded")
        plt.plot(cond.cpu().detach().numpy(), model_base, "-o", label="w/ Physics Encoded")
        plt.legend()
        plt.ylabel("Relative Error [%]")
        plt.xlabel("SNR [dB]")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"error_ablation_{k}_{k_iid}.svg")
        plt.show()
    res_per_size = np.asarray(res_per_size)
    plt.semilogx(res_per_size[:, 0], res_per_size[:, 1], "--x", label="w/o Physics Encoded")
    plt.semilogx(res_per_size[:, 0], res_per_size[:, 2], "-o", label="w/ Physics Encoded")
    plt.legend()
    plt.ylabel("Mean Relative Error [%]")
    plt.xlabel(r"$N_{\mathcal{D}}$")
    plt.grid()
    plt.tight_layout()
    plt.ylim([0, 25])
    plt.savefig(f"error_ablation_samples_{k_iid}.svg")
    plt.show()