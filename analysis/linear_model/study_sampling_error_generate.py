import matplotlib.pyplot as plt
import numpy as np
import torch

from pyresearchutils.seed import torch
from signal_models import LinearGaussianProblem
import pyresearchutils as pru
from tqdm import tqdm

if __name__ == '__main__':
    n = 16
    m = 4
    k = 64
    batch_size = 512
    mc_try = 1000
    n_samples_base = 32000 * 8

    for n_samples_div in [4]:
        n_samples = n_samples_base // n_samples_div
        print(n_samples)
        pru.set_seed(42)
        linear_problem = LinearGaussianProblem(n, m, k, alpha_rho=6, beta_rho=2.5, minimal_snr=-2, maximal_snr=2,
                                               snr_points=2)
        prior_score = linear_problem.get_optimal_prior_score()
        likelihood_score = linear_problem.get_optimal_likelihood_score()

        res = np.zeros([len(linear_problem.get_condition_list()), mc_try, 12])
        for i, snr_target in enumerate(linear_problem.get_condition_list()):
            bcrb, fim, pfim = linear_problem.bcrb(snr_target, k)
            for j in tqdm(range(mc_try)):
                dataset = linear_problem.get_dataset(n_samples, cond=snr_target)

                dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

                _bfim = torch.zeros(m, m, device=pru.get_working_device())
                _bfim_decomposition = torch.zeros(m, m, device=pru.get_working_device())
                count = 0
                c_f = 0
                c_p = 0
                c_b = 0
                c_mp = 0
                with torch.no_grad():
                    for x, theta, snr in dl:
                        x = x.to(pru.get_working_device())
                        theta = theta.to(pru.get_working_device())
                        snr = snr.to(pru.get_working_device())
                        _prior_score = prior_score(theta)
                        _likelihood_score = likelihood_score(x, theta, snr, split_iid=True)
                        _posterior_score = torch.sum(_likelihood_score, dim=1) + _prior_score

                        lbfimb = _posterior_score.unsqueeze(dim=-1) @ _posterior_score.unsqueeze(dim=-2)
                        lbfimm = torch.sum(_likelihood_score.unsqueeze(dim=-1) @ _likelihood_score.unsqueeze(dim=-2),
                                           dim=1)
                        lbfimp = _prior_score.unsqueeze(dim=-1) @ _prior_score.unsqueeze(dim=-2)

                        lbfimmp = lbfimm + lbfimp

                        c_f = max(torch.linalg.norm(lbfimm, ord=2, dim=(1, 2)).max().item(), c_f)
                        c_p = max(torch.linalg.norm(lbfimp, ord=2, dim=(1, 2)).max().item(), c_p)
                        c_b = max(torch.linalg.norm(lbfimb, ord=2, dim=(1, 2)).max().item(), c_b)
                        c_mp = max(torch.linalg.norm(lbfimmp, ord=2, dim=(1, 2)).max().item(), c_mp)

                        _bfim_current = torch.mean(
                            lbfimb,
                            dim=0)
                        _efim_current = torch.mean(
                            lbfimm,
                            dim=0)
                        _pfim_current = torch.mean(lbfimp,
                                                   dim=0)

                        w_past = count / (count + _prior_score.shape[0])
                        w_current = _prior_score.shape[0] / (count + _prior_score.shape[0])
                        _bfim = w_current * _bfim_current + w_past * _bfim
                        _bfim_decomposition = w_current * (_efim_current + _pfim_current) + w_past * _bfim_decomposition
                        count += _prior_score.shape[0]
                    re = (100 * torch.linalg.norm(torch.linalg.inv(_bfim) - bcrb, ord=2) / torch.linalg.norm(bcrb,
                                                                                                             ord=2)).item()
                    re_dec = (100 * torch.linalg.norm(torch.linalg.inv(_bfim_decomposition) - bcrb,
                                                      ord=2) / torch.linalg.norm(bcrb,
                                                                                 ord=2)).item()

                    re_fim = (100 * torch.linalg.norm(torch.linalg.inv(bcrb) - _bfim, ord=2) / torch.linalg.norm(
                        torch.linalg.inv(bcrb),
                        ord=2)).item()
                    re_dec_fim = (100 * torch.linalg.norm(torch.linalg.inv(bcrb) - _bfim_decomposition,
                                                          ord=2) / torch.linalg.norm(torch.linalg.inv(bcrb),
                                                                                     ord=2)).item()
                    lbfim_norm = torch.linalg.norm(_bfim, ord=2).item()
                    kappa = lbfim_norm * torch.linalg.norm(torch.linalg.inv(_bfim), ord=2).item()

                    res[i, j, 0] = re
                    res[i, j, 1] = re_dec
                    res[i, j, 2] = torch.trace(torch.linalg.inv(bcrb)).item()
                    res[i, j, 3] = torch.linalg.norm(torch.linalg.inv(bcrb), ord=2).item()
                    res[i, j, 4] = c_f
                    res[i, j, 5] = c_p
                    res[i, j, 6] = c_b
                    res[i, j, 7] = kappa
                    res[i, j, 8] = lbfim_norm
                    res[i, j, 9] = c_mp

                    res[i, j, 10] = re_fim
                    res[i, j, 11] = re_dec_fim
                    # res[i, j, 9] = torch.linalg.norm(bcrb, ord=2).item()
        if k!=10:
            pickle_path = f"sampling_error_analysis_{n_samples}_{m}_{k}_snr.pkl"
        else:
            pickle_path = f"sampling_error_analysis_{n_samples}_{m}_snr.pkl"
        with open(pickle_path, "wb") as f:
            import pickle

            pickle.dump(res, f)
