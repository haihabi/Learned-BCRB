from sympy.abc import kappa

from signal_models import LinearGaussianProblem

import numpy as np

def n_zero(u, d, r):
    s = (u + np.log(8 * d)) / 3
    return s * (r * d + 1)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    fim_error=False
    cmap = plt.cm.get_cmap('viridis')  # Choose any colormap you like
    plt.rcParams.update({'font.size': 12})
    SAMPLING_SIZE = [16, 32, 64, 128, 256]
    res_list = []
    ind = 1
    snr_name="high_snr" if ind == 1 else "low_snr"
    for size in SAMPLING_SIZE:
        size *= 1000
        with open(f"sampling_error_analysis_{size}_4_snr.pkl", "rb") as f:
            res = pickle.load(f)
        trace = res[ind, 0, 2]
        norm = res[ind, 0, 3]
        c_f = res[ind, :, 4]
        c_p = res[ind, :, 5]
        c_b = res[ind, :, 6]
        d = trace / norm
        if fim_error:
            m = np.mean(res[ind, :, -2])
            std = np.std(res[ind, :, -2])
        else:
            m = np.mean(res[ind, :, 0])
            std = np.std(res[ind, :, 0])
        if fim_error:
            m_mp = np.mean(res[ind, :, -1])
            std_mp = np.std(res[ind, :, -1])
        else:
            m_mp = np.mean(res[ind, :, 1])
            std_mp = np.std(res[ind, :, 1])
        # m_mp = np.mean(res[ind, :, -1])
        # std_mp = np.std(res[ind, :, -1])
        lbfim_norm = res[ind, :, 8]

        max_index = np.argmin(c_b)
        kappa = res[ind, max_index, 7]
        res_list.append([size, m, std, d, trace, c_b[max_index],c_p[max_index],c_f[max_index],m_mp,std_mp,kappa,lbfim_norm[max_index]])
    res = np.array(res_list)
    p = 0.9

    u = -np.log(1 - p)
    d = res[:, 3]
    n_ds = res[:, 0]
    c_b = res[:, 5]
    c_p = res[:, 6]
    c_f = res[:, 7]
    trace = res[:, 4]


    rr = n_zero(u, d,  c_b  / trace) / n_ds
    rr_mp = n_zero(u, d, (c_p +c_f ) / trace) / n_ds
    # print(rr)
    upper_bound = 100 * np.sqrt(3*rr)
    upper_bound_mp = 100 *  np.sqrt(3*rr_mp)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(res[:, 0], upper_bound, label="Posterior")
    ax1.plot(res[:, 0], upper_bound_mp, "--", label="Measurement Prior")

    ax1.set_ylabel("Theoretical Bound [%]")
    ax1.grid()
    ax1.legend()
    ax2.errorbar(res[:, 0], res[:, 1], yerr=res[:, 2], fmt='o', label="Posterior")
    ax2.errorbar(res[:, 0], res[:, 8], yerr=res[:, 9], fmt='x', label="Measurement Prior")


    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    plt.xlabel("$N_{\mathcal{D}}^c$")
    plt.ylabel("Empirical Error [%]")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if fim_error:
        plt.savefig(f"sampling_error_analysis_vs_n_samples_fim_{snr_name}.svg")
    else:
        plt.savefig(f"sampling_error_analysis_vs_n_samples_{snr_name}.svg")
    plt.show()

    n = 16
    color_list=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive",
     "tab:cyan"]
    for icolor,m in enumerate([1,4, 8, 16]):
        with open(f"sampling_error_analysis_64000_{m}_snr.pkl", "rb") as f:
            res = pickle.load(f)

        _res = res[ind, :, :]
        if fim_error:
            count, bins = np.histogram(_res[:, -1], bins=20,density=True)
        else:
            count, bins = np.histogram(_res[:, 1], bins=20,density=True)
        # count, bins = np.histogram(_res[:, -1], bins=20)
        delta_x = np.mean(np.diff(bins))
        plt.plot(bins[:-1], count , color=color_list[icolor],
                 label=r"Measurement-Prior $d_\theta$" + f"={m}")
        if fim_error:
            count, bins = np.histogram(_res[:, -2], bins=20,density=True)
        else:
            count, bins = np.histogram(_res[:, 0], bins=20,density=True)
        delta_x = np.mean(np.diff(bins))
        plt.plot(bins[:-1], count , "--", color=color_list[icolor],
                 label=r"Posterior  $d_\theta$" + f"={m}")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Relative Frequency")
    plt.grid()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0)
    plt.tight_layout()
    if fim_error:
        plt.savefig(f"sampling_error_analysis_different_m_fim_{snr_name}.svg")
    else:
        plt.savefig(f"sampling_error_analysis_different_m_{snr_name}.svg")
    plt.show()
    for m in [4]:
        with open(f"sampling_error_analysis_64000_{m}_snr.pkl", "rb") as f:
            res = pickle.load(f)
        linear_problem = LinearGaussianProblem(n, m, 1, alpha_rho=6, beta_rho=2.5, minimal_snr=-2, maximal_snr=2,
                                               snr_points=2)
        print("a")
        for i, snr_target in enumerate(linear_problem.get_condition_list()):
            _res = res[i, :, :]
            snr_target = round(snr_target.item(), 1)
            if fim_error:
                count, bins = np.histogram(_res[:, -1], bins=20,density=True)
            else:
                count, bins = np.histogram(_res[:, 1], bins=20,density=True)
            delta_x = np.mean(np.diff(bins))
            plt.plot(bins[:-1], count,color=color_list[i], label=f"Measurement Prior SNR={snr_target} dB")
            if fim_error:
                count, bins = np.histogram(_res[:, -2], bins=20,density=True)
            else:
                count, bins = np.histogram(_res[:, 0], bins=20,density=True)
            delta_x = np.mean(np.diff(bins))
            plt.plot(bins[:-1], count, "--",color=color_list[i], label=f"Posterior SNR={round(snr_target)}dB")
        plt.xlabel("Relative Error (%)")
        plt.ylabel("Relative Frequency")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        if fim_error:
            plt.savefig("sampling_error_analysis_different_snr_fim.svg")
        else:
            plt.savefig("sampling_error_analysis_different_snr.svg")
        plt.show()

        m=4
        res_list=[]
        for k in [1,2,4,8,10,16,64,100]:
            if k==10:
                path_location=f"sampling_error_analysis_64000_4_snr.pkl"
            else:
                path_location=f"sampling_error_analysis_64000_4_{k}_snr.pkl"
            with open(path_location, "rb") as f:
                res = pickle.load(f)
            linear_problem = LinearGaussianProblem(n, 4, k, alpha_rho=6, beta_rho=2.5, minimal_snr=-5, maximal_snr=2,
                                                   snr_points=2)
            trace = res[ind, 0, 2]
            norm = res[ind, 0, 3]
            c_f = res[ind, :, 4]
            c_p = res[ind, :, 5]
            c_b = res[ind, :, 6]
            d = trace / norm
            if fim_error:
                m = np.mean(res[ind, :, -2])
                std = np.std(res[ind, :, -2])
            else:
                m = np.mean(res[ind, :, 0])
                std = np.std(res[ind, :, 0])
            if fim_error:
                m_mp = np.mean(res[ind, :, -1])
                std_mp = np.std(res[ind, :, -1])
            else:
                m_mp = np.mean(res[ind, :, 1])
                std_mp = np.std(res[ind, :, 1])
            # m_mp = np.mean(res[ind, :, -1])
            # std_mp = np.std(res[ind, :, -1])
            lbfim_norm = res[ind, :, 8]

            max_index = np.argmin(c_b)
            kappa = res[ind, max_index, 7]
            res_list.append(
                [k, m, std, d, trace, c_b[max_index], c_p[max_index], c_f[max_index], m_mp, std_mp, kappa,
                 lbfim_norm[max_index]])
        res = np.array(res_list)
        p = 0.9

        u = -np.log(1 - p)
        d = res[:, 3]
        n_ds = 64000
        c_b = res[:, 5]
        c_p = res[:, 6]
        c_f = res[:, 7]
        trace = res[:, 4]

        rr = n_zero(u, d, c_b / trace) / n_ds
        rr_mp = n_zero(u, d, (c_p + c_f) / trace) / n_ds
        # print(rr)
        upper_bound = 100 * np.sqrt(3 * rr)
        upper_bound_mp = 100 * np.sqrt(3 * rr_mp)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(res[:, 0], upper_bound, label="Posterior")
        ax1.plot(res[:, 0], upper_bound_mp, "--", label="Measurement Prior")

        ax1.set_ylabel("Theoretical Bound [%]")
        ax1.grid()
        ax1.legend()
        ax2.errorbar(res[:, 0], res[:, 1], yerr=res[:, 2], fmt='o', label="Posterior")
        ax2.errorbar(res[:, 0], res[:, 8], yerr=res[:, 9], fmt='x', label="Measurement Prior")
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        plt.xlabel("$m_{iid}$")
        plt.ylabel("Empirical Error [%]")
        plt.xscale("log")
        # plt.legend()
        plt.grid()
        plt.tight_layout()

        if fim_error:
            plt.savefig(f"sampling_error_analysis_vs_m_iid_fim_{snr_name}.svg")
        else:
            plt.savefig(f"sampling_error_analysis_vs_m_iid_{snr_name}.svg")
        plt.show()