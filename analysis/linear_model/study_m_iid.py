import matplotlib.pyplot as plt

import pyresearchutils as pru
import numpy as np
import matplotlib.ticker as mticker
from analysis.helpers import generate_bound_function, load_wandb_run, run_reference, bcrb_relative_error


def run_analysis_k_study(in_run_name, number_of_samples=128000, batch_size=512, compute_optimal=False,
                         use_optimal_score=False, k=None,remove_mean=False):
    pru.set_seed(42)
    model, problem, param,_ = load_wandb_run(in_run_name)
    compute_bcrb, sigma_array = generate_bound_function(problem, model, score_model_type=param.score_model_type,
                                                        number_of_samples=number_of_samples,
                                                        compute_optimal=compute_optimal, remove_mean=remove_mean)
    k = param.k if k is None else k
    print(k)
    bcrb, _, bound_array_function = run_reference(problem, k, number_of_samples / batch_size, batch_size,
                                                  pru.get_working_device(),
                                                  m=param.m,
                                                  in_run_est=False)

    def gbcrb_array_function(in_k, index=0):
        return np.stack(
            [compute_bcrb(in_k, cond.item(), opt=use_optimal_score, debug=True)[index] for cond in
             problem.get_condition_list()])

    score_bcrb = gbcrb_array_function(k)
    bcrb = bcrb.cpu().numpy()
    re_array = bcrb_relative_error(bcrb, score_bcrb)
    re_mean = np.mean(re_array)
    re_max = np.max(re_array)
    return re_mean, re_max, re_array, problem.get_condition_list(), bound_array_function, gbcrb_array_function


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 12})
    k_likelihood_run_list = ["revived-armadillo-636", "swift-music-640", "woven-glade-641", "dauntless-serenity-642",
                             "worldly-oath-624"]
    k_posterior_run_list = ["whole-pine-633", "sleek-plasma-637", "happy-vortex-638", "lemon-wind-639",
                            "charmed-sun-628"]

    k_likelihood_run_list = ["fine-mountain-1528", "swift-glade-1566", "winter-sun-1592", "dazzling-microwave-1568",
                             "denim-leaf-1600"]
    k_posterior_run_list = ["swept-sun-1551", "decent-aardvark-1552", "graceful-oath-1553", "wobbly-dust-1554",
                            "still-sunset-1555"]
    error_list = []
    k = np.arange(1, 6)
    print(k)
    for run_lik, run_post, _k in zip(k_likelihood_run_list, k_posterior_run_list, k):
        mean_lik, max_lik, _, cond, ref_func, gbcrb_func = run_analysis_k_study(run_lik)

        re_matrix = []
        k_array = np.logspace(1, 7, 7)
        for k in k_array:
            re_array = bcrb_relative_error(ref_func(k).cpu().detach().numpy(), gbcrb_func(k))
            re_matrix.append(re_array)
        re_matrix = np.array(re_matrix)
        print(re_matrix.shape)
        plt.semilogx(k_array, re_matrix[:, 0], label=f"SNR={cond[0]}")
        plt.semilogx(k_array, re_matrix[:, -1], label=f"SNR={cond[-1]}")
        plt.semilogx(k_array, np.mean(re_matrix, axis=1), label="Mean")
        plt.semilogx(k_array, np.max(re_matrix, axis=1), label="Max")
        plt.legend()
        plt.xlabel(r"$m_{iid}$")
        plt.ylabel("Relative Error [%]")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"k_study_extrapolation_{_k}.svg")
        plt.show()

        mean_post, max_post, _, _, _, _ = run_analysis_k_study(run_post)
        error_list.append((mean_lik, max_lik, mean_post, max_post))
        print(f"Likelihood:{mean_lik},{max_lik} Posterior:{mean_post},{max_post}")
    error_array = np.array(error_list)
    x=[1,2,3,4,5]
    width=0.2
    plt.bar(np.asarray(x)-width/2, error_array[:, 0],width=width, color="blue", label="Measurement-Prior")
    plt.bar(np.asarray(x)+width/2, error_array[:, 2],width=width, color="red", label="Posterior")
    plt.xlabel(r"$n_{iid}$")
    plt.ylabel("Mean Relative Error [%]")
    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig("k_study.svg")

    plt.show()
