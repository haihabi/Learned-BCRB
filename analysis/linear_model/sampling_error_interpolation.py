import matplotlib.pyplot as plt

from analysis.linear_model.study_sampling_error_plot import n_zero
from analysis.score2fim_accumulator import Score2FIMAccumulator
from pyresearchutils.seed import torch
from pyresearchutils.signal_processing.metric import torch
from signal_models import LinearGaussianProblem
import pyresearchutils as pru
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    n_samples=64000
    dp=4
    dx=10

    m_iid=10

    linear_problem = LinearGaussianProblem(dx, dp, m_iid, alpha_rho=6, beta_rho=2.5, minimal_snr=-5, maximal_snr=2,snr_points=2)

    plt.rcParams.update({'font.size': 12})
    for snr_target in linear_problem.get_condition_list():
        efim_list=[]
        efim_c_max_list=[]
        pfim_list=[]
        pfim_c_max_list=[]
        for _ in tqdm(range(1000)):
            efim = Score2FIMAccumulator()
            pfim = Score2FIMAccumulator()
            dataset=linear_problem.get_dataset(n_samples, cond=snr_target)
            data_loader=torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)

            for x, theta, snr in data_loader:
                x = x.to(pru.get_working_device())
                theta = theta.to(pru.get_working_device())
                snr = snr.to(pru.get_working_device())
                prior_score = linear_problem.get_optimal_prior_score()(theta)
                likelihood_score = linear_problem.get_optimal_likelihood_score()(x, theta, snr, split_iid=True)
                # posterior_score = torch.sum(likelihood_score, dim=1) + prior_score
                efim.accumulate(likelihood_score)
                pfim.accumulate(prior_score)

            efim_fim = efim.get_fim()
            pfim_fim = pfim.get_fim()
            efim_list.append(efim_fim)
            pfim_list.append(pfim_fim)
            efim_c_max_list.append(efim.c_max)
            pfim_c_max_list.append(pfim.c_max)

        bcrb,efim_ref,pfim_ref=linear_problem.bcrb(snr_target, 1)
        efim_list=torch.stack(efim_list)
        pfim_list=torch.stack(pfim_list)
        re_list=[]
        n_iid_array=[1,4,8,10,16,32,64,128,1024]
        p = 0.9

        u = -np.log(1 - p)
        for n_iid in n_iid_array:

            bfim= n_iid * efim_list + pfim_list
            bfim_ref= n_iid * efim_ref.unsqueeze(dim=0)  + pfim_ref.unsqueeze(dim=0)
            c_m = torch.stack(efim_c_max_list).max().item()
            c_p = torch.stack(pfim_c_max_list).max().item()
            d= (torch.trace(bfim[0,:,:])/torch.linalg.norm(bfim[0,:,:],ord=2)).item()
            bound=torch.sqrt(3*n_zero(u,d ,(c_m+c_p/n_iid)/(torch.trace(efim_ref)+torch.trace(pfim_ref)/n_iid))/n_samples)

            re=100*(torch.linalg.norm(bfim-bfim_ref,ord=2,dim=(1,2))/torch.linalg.norm(bfim_ref,ord=2,dim=(1,2)))
            re_list.append([torch.mean(re).item(),torch.std(re).item(),100*bound.item()])
        re_list = np.asarray(re_list)
        min_point=np.min(np.asarray(re_list[:, 0])-np.asarray(re_list[:, 1]))
        max_point=np.max(np.asarray(re_list[:, 0])+np.asarray(re_list[:, 1]))


        fig, ax1 = plt.subplots()
        color = 'tab:red'
        min_point_t=np.min(re_list[:,-1])
        max_point_t=np.max(re_list[:,-1])
        ax1.plot(n_iid_array,re_list[:,-1],color=color)
        ax1.set_ylabel("Theoretical Error (%)", color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(min_point_t-0.2*min_point_t,max_point_t*1.02)

        color = 'tab:blue'
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        ax2.errorbar(n_iid_array, re_list[:, 0], yerr=re_list[:, 1],color=color)
        ax2.set_ylabel("Empirical Error (%)", color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(min_point*0.95,max_point*1.055)


        plt.xscale("log")
        ax1.set_xlabel(r"$n_{iid}$")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"sampling_error_analysis_vs_n_iid_{snr_target}.svg")
        plt.show()


        # plt.xscale("log")
        # plt.grid()
        # plt.xlabel(r"$n_{iid}$")
        #
        # plt.tight_layout()
        # plt.savefig(f"sampling_error_analysis_vs_n_iid_{snr_target}.svg")
        # plt.show()



