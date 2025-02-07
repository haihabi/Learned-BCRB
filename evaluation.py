import torch
from analysis.score2fim_accumulator import ConditionedScore2FIMAccumulator
from utils.init_run import init_run
from utils.ema import load_ema_state_dict
from utils.constants import SCORE_LAST_EMA
import pyresearchutils as pru
from utils.config import ScoreModelType
from tqdm import tqdm
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data_loader, device, model, param, problem, val_loader = init_run(False,
                                                                      is_eval=True)  # Initialize the run parameters for evaluation
    load_ema_state_dict(device, SCORE_LAST_EMA, model)  # Load the EMA state dictionary

    cs2fim_acc = ConditionedScore2FIMAccumulator(problem.get_condition_list())
    if problem.has_score:
        cs2fim_acc_opt = ConditionedScore2FIMAccumulator(problem.get_condition_list())
        ops = problem.get_optimal_prior_score()
        ols = problem.get_optimal_likelihood_score()


        def optimal_posterior(x, theta, cond):
            return ols(x, theta, cond) + ops(theta)
    for x, theta, cond in tqdm(data_loader):
        x = x.to(pru.get_working_device())
        theta = theta.to(pru.get_working_device())
        cond = cond.to(pru.get_working_device())
        if param.score_model_type == ScoreModelType.Posterior:
            score = model(x, theta, cond)
            cs2fim_acc.accumulate(score, cond)
            if problem.has_score:
                optimal_posterior_score = optimal_posterior(x, theta, cond)
                cs2fim_acc_opt.accumulate(optimal_posterior_score, cond)
        else:
            raise NotImplementedError
            score = model.likelihood_score(x, theta, cond, sum_over_iid=False)
            prior_score = model.prior_score(theta)

    lbfim = cs2fim_acc.get_fim()  # Get the Bayesian Fisher Information Matrix
    lbcrb = torch.linalg.inv(lbfim)  # Invert the Fisher Information Matrix
    cond_array = problem.get_condition_list()

    if problem.has_bcrb:
        bcrb = torch.stack([problem.bcrb(c)[0] for c in cond_array])
        re = torch.linalg.norm(bcrb - lbcrb, ord=2, dim=(1, 2)) / torch.linalg.norm(bcrb, ord=2, dim=(1, 2))

        plt.plot(cond_array.cpu().detach().numpy(), 100 * re.cpu().detach().numpy())
        plt.grid()
        plt.ylabel('Relative Error (%)')
        plt.xlabel('SNR[dB]')

        plt.grid()
        plt.show()

        plt.plot(cond_array.cpu().detach().numpy(), torch.diagonal(bcrb, dim1=1, dim2=2).sum(-1).cpu().detach().numpy())
        plt.plot(cond_array.cpu().detach().numpy(),
                 torch.diagonal(lbcrb, dim1=1, dim2=2).sum(-1).cpu().detach().numpy())
        plt.yscale('log')
        plt.show()
        pass
    else:

        if problem.has_score:
            lbcrb_opt = torch.linalg.inv(cs2fim_acc_opt.get_fim())
            re = torch.linalg.norm(lbcrb_opt - lbcrb, ord=2, dim=(1, 2)) / torch.linalg.norm(lbcrb_opt, ord=2,
                                                                                             dim=(1, 2))
            plt.plot(cond_array.cpu().detach().numpy(), 100 * re.cpu().detach().numpy())
            plt.grid()
            plt.ylabel('Relative Error (%)')
            plt.xlabel('SNR[dB]')
            plt.show()

        plt.plot(cond_array.cpu().detach().numpy(),
                 torch.diagonal(lbcrb, dim1=1, dim2=2).sum(-1).cpu().detach().numpy(), label="Learned Score")
        if problem.has_score:
            plt.plot(cond_array.cpu().detach().numpy(),
                     torch.diagonal(lbcrb_opt, dim1=1, dim2=2).sum(-1).cpu().detach().numpy(), label="True Score")
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.xlabel('SNR[dB]')
        plt.ylabel('Trace(LBCRB)')
        plt.show()
        print("No BCRB available")
