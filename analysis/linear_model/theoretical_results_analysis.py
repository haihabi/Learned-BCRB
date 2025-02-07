import matplotlib.pyplot as plt
import numpy as np

from analysis.helpers import load_wandb_run


def filter_none(results):
    results = np.array(results)
    results[results == None] = np.nan
    results = results.astype(float)
    ind = np.logical_not(np.any(np.isnan(results), axis=1))
    results = results[ind, :]
    return results


if __name__ == '__main__':
    run_name = "balmy-shadow-98"
    snr = 20.0
    model, problem, param, run = load_wandb_run(run_name)
    data = run.history()
    base = f"cond_{str(snr)}/"
    results = [[d.get(base + n) for n in ["score_norm2_ema_lik", "efim_re", "efim_nd"]] for d in data]
    results_metric = [[d.get(base + n) for n in ["l_tilde", "C_approx", "fim_norm"]] for d in data]
    results = filter_none(results)
    results_metric = filter_none(results_metric)
    # print(images.shape)
    # print((images[:,2]-results_metric[:,2])/images[:,2])
    # print()
    bound = (results[:, 0] + 2 * np.sqrt(results[:, 2] * results[:, 0])) / results[:, 2]
    # print("a")
    l_approx = np.abs(results_metric[:, 0] + results_metric[:, 1])

    plt.plot(results[:, 0],label="True Loss")
    # plt.plot(images[:, 1])
    plt.plot(l_approx,label="Approx Loss")
    # plt.plot(images[:, 0])
    plt.legend()
    plt.grid()
    plt.show()
    print(np.min(results[:,1]))
    plt.plot(results[:, 1], label="RE")
    plt.plot((l_approx/results_metric[:,2]+2*np.sqrt(l_approx/results_metric[:,2]))*100, label="Suggested Metric")
    plt.plot(0.5 + bound * 100, label="Bound Exact")
    plt.legend()
    plt.ylim([0, 50])
    # plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("RE [%]")
    plt.grid()
    plt.tight_layout()

    plt.show()
