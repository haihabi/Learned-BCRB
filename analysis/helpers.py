import os
from argparse import Namespace

import numpy as np
import torch
import wandb
from tqdm import tqdm

import pyresearchutils as pru
from analysis.neural_mmse import train_mmse_estimator
from analysis.score2fim_accumulator import Score2FIMAccumulator
from utils.constants import PROJECT
from neural_networks.base_score_model import BaseScoreModel
from neural_networks.basic_blocks.non_linearity_factory import NonLinearityType
from signal_models.base_problem import BaseProblem
from signal_models.quantization.base_quantization import infmean
from utils.config import ScoreModelType, ScoreLossType, SignalModelType, NoiseType
from utils.builder import param2score_nn, param2problem
from utils.ema import load_ema_state_dict


def load_wandb_run(run_name, project_name=f"HVH/{PROJECT}", file_name="model_ema_last.pt",
                   file_model="model_problem.pt"):
    api = wandb.Api()
    runs = api.runs(project_name)
    for run in runs:
        if run.name == run_name:
            config = run.config
            device = pru.get_working_device()
            enums = [ScoreLossType, ScoreModelType, SignalModelType, NonLinearityType,
                     NonLinearityType, NoiseType]
            names = ["loss_type", "score_model_type", "signal_model", "non_linearity",
                     "non_linearity_prior", "noise_type"]
            for name, enum in zip(names, enums):
                if config.get(name) is not None:
                    config[name] = enum[config[name].split(".")[-1]]
                else:
                    config[name] = None
            if config.get("rho_cov") is None:
                config["rho_cov"] = 0
            if config.get("inject") is None:
                config["inject"] = False
            if config.get("output_bias") is None:
                config["output_bias"] = False
            if config.get("output_bias_prior") is None:
                config["output_bias_prior"] = True
            if config.get("non_linearity_normalization") is None:
                config["non_linearity_normalization"] = True
            if config.get("spacing") is None:
                config["spacing"] = 0.1
            if config.get("is_random_phase") is None:
                config["is_random_phase"] = False
            if config.get("output_rescale") is None:
                config["output_rescale"] = False
            param = Namespace(**config)

            problem = param2problem(param)
            if os.path.isfile(file_model):
                os.remove(file_model)
            run.file(file_model).download()
            problem.load_state_dict(torch.load(file_model, map_location=device, weights_only=True))
            model = param2score_nn(device, param, problem)

            if os.path.isfile(file_name):
                os.remove(file_name)
            run.file(file_name).download()
            load_ema_state_dict(device, file_name, model)
            model = model.eval()
            if config.get("optimal_prior"):
                model.set_optimal_prior(problem.get_optimal_prior_score())
            return model, problem, param, run


def run_reference(in_qlp: BaseProblem, k_iid_samples, n_iter, batch_size, device, m=1, in_run_est=False):
    sigma_array = in_qlp.get_condition_list()
    map_error = 0
    mmse_error = 0
    if not in_qlp.has_bcrb:
        efim = 0
        ecrb = 0

        count = 0
        for j in tqdm(range(n_iter)):
            current_scale = batch_size / (count + batch_size)
            past_scale = count / (count + batch_size)
            count += batch_size
            param = in_qlp.obq.sample_parameters(batch_size, m).to(device)
            if in_run_est:
                signal = in_qlp.obq.generate_signal(param)
                x = signal.unsqueeze(dim=-2) + torch.randn([batch_size, 1, k_iid_samples, signal.shape[-1]]).to(
                    device) * sigma_array.reshape(
                    [1, -1, 1, 1])  # Measurement with noise
                x = in_qlp.obq.quantization(x)
                param_hat_map = in_qlp.obq.map_estimator(x, sigma_array, m)
                mse_array = torch.mean((param.unsqueeze(dim=-1) - param_hat_map) ** 2, dim=0).detach()
                map_error = mse_array.flatten() * current_scale + map_error * past_scale

                mse_array = mmse_error = None

            fisher_array = in_qlp.obq.fisher_one_bit_linear(param, sigma_array)
            efim = infmean(fisher_array, 1) * current_scale + efim * past_scale
            ecrb = infmean(1 / (fisher_array + 1e-8), 1) * current_scale + ecrb * past_scale
        bcrb = 1 / (k_iid_samples * efim + in_qlp.obq.prior_fim(None))
    else:
        def bound_array_function(in_k_iid, index=0):
            return torch.stack([in_qlp.bcrb(cond, in_k_iid)[index] for cond in sigma_array], dim=0)

        bcrb = bound_array_function(k_iid_samples)

    return bcrb, (map_error, mmse_error), bound_array_function


def generate_bound_function(in_problem,
                            in_model,
                            score_model_type: ScoreModelType = ScoreModelType.PriorLikelihoodSplitIID,
                            split_iid_in_eval=True,
                            number_of_samples=64000,
                            compute_optimal=False,
                            remove_mean=False):
    bcrb_dict = {}
    optimal_prior_score = optimal_likelihood_score = None
    if compute_optimal:
        optimal_prior_score = in_problem.get_optimal_prior_score()
        optimal_likelihood_score = in_problem.get_optimal_likelihood_score()
    for cond_loop in tqdm(in_problem.get_condition_list()):
        dataset = in_problem.get_dataset(number_of_samples, cond=cond_loop, iid_dataset=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
        bfim, fim, fim_opt, norm_base_lik, norm_base_prior, norm_delta_lik, norm_delta_prior, prior_fim, prior_fim_opt = gbcrb_bound_per_cond(
            compute_optimal, data_loader, in_model, optimal_likelihood_score, optimal_prior_score, score_model_type,
            split_iid_in_eval, remove_mean)
        if score_model_type == ScoreModelType.Posterior:
            bcrb_dict.update({cond_loop.item(): bfim.detach().cpu().numpy()})
        else:
            if compute_optimal:
                print(f"Score Error Analysis @ Condition: {cond_loop.item()}")
                print("Prior Score Error", 100 * torch.cat(norm_delta_prior).mean() / torch.cat(norm_base_prior).mean())
                print("Likelihood Score Error",
                      100 * torch.cat(norm_delta_lik).mean() / torch.cat(norm_base_lik).mean())
                bcrb_dict.update(
                    {cond_loop.item(): (
                        fim.detach().cpu().numpy(), prior_fim.detach().cpu().numpy(), fim_opt.detach().cpu().numpy(),
                        prior_fim_opt.detach().cpu().numpy())})
            else:
                bcrb_dict.update(
                    {cond_loop.item(): (
                        fim.detach().cpu().numpy(), prior_fim.detach().cpu().numpy())})

    ###############
    # Plot the images
    ###############
    if score_model_type in [ScoreModelType.PriorLikelihoodSplitIID,
                            ScoreModelType.PriorLikelihoodSplitIIDModelInformed]:
        def _compute_bcrb(in_k_iid, in_sigma, debug=False, opt=False):
            if compute_optimal:
                _fim, _prior_fim, _fim_opt, _prior_fim_opt = bcrb_dict[in_sigma]
            else:
                _fim, _prior_fim = bcrb_dict[in_sigma]
            if opt:
                if not compute_optimal:
                    raise Exception("No optimal bound available")
                _fim = _fim_opt
                _prior_fim = _prior_fim_opt
            if debug:
                return np.linalg.inv(in_k_iid * _fim + _prior_fim), in_k_iid * _fim, _prior_fim
            return np.linalg.inv(in_k_iid * _fim + _prior_fim)
    elif score_model_type == ScoreModelType.PriorLikelihoodSplit:
        def _compute_bcrb(k_iid, in_sigma, debug=False, opt=False):
            if compute_optimal:
                _fim, _prior_fim, _fim_opt, _prior_fim_opt = bcrb_dict[in_sigma]
            else:
                _fim, _prior_fim = bcrb_dict[in_sigma]
            if opt:
                if not compute_optimal:
                    raise Exception("No optimal bound available")
                _fim = _fim_opt
                _prior_fim = _prior_fim_opt
            if debug:
                return np.linalg.inv(_fim + _prior_fim), _fim, _prior_fim
            return np.linalg.inv(_fim + _prior_fim)
    elif score_model_type == ScoreModelType.Posterior:
        def _compute_bcrb(k_iid, in_sigma, debug=False, opt=False):
            if compute_optimal:
                _fim = bcrb_dict[in_sigma]
            else:
                _fim = bcrb_dict[in_sigma]
            if opt:
                if not compute_optimal:
                    raise Exception("No optimal bound available")
                _fim = bcrb_dict[in_sigma]
            if debug:
                return np.linalg.inv(_fim), _fim, None
            return np.linalg.inv(_fim)
    else:
        raise Exception("Unknown score model type")

    return _compute_bcrb, bcrb_dict.keys()


def lbcrb_bound_per_cond(in_dataloader, score_function_dict, split_iid_in_eval=False):
    results = {k: Score2FIMAccumulator() if len(v) == 1 else (
        Score2FIMAccumulator(split_iid_in_eval=split_iid_in_eval), Score2FIMAccumulator()) for k, v in
               score_function_dict.items()}
    with torch.no_grad():
        for x, theta, cond in in_dataloader:
            x = x.to(pru.get_working_device())
            theta = theta.to(pru.get_working_device())
            cond = cond.to(pru.get_working_device())
            for score_type, scores in score_function_dict.items():
                results_acc = results[score_type]
                if len(scores) == 1:
                    score = scores(x, theta, cond)
                    results_acc.accumulate(score)
                else:
                    likelihood_score = scores[0](x, theta, cond, sum_over_iid=False)
                    prior_score = scores[1](theta)
                    results_acc[0].accumulate(likelihood_score)
                    results_acc[1].accumulate(prior_score)
    return results


def lbcrb_bound(in_problem,
                split_iid_in_eval=False,
                number_of_samples=64000,
                batch_size=1024,
                compute_optimal=False,
                model: BaseScoreModel = None):
    score_function_dict = {}
    if compute_optimal:
        optimal_prior_score = in_problem.get_optimal_prior_score()
        optimal_likelihood_score = in_problem.get_optimal_likelihood_score()
        if optimal_likelihood_score is None or optimal_prior_score is None:
            raise Exception("Optimal scores are not available")
        score_function_dict.update({"optimal": (optimal_likelihood_score, optimal_prior_score)})
    if model is not None:
        if model.score_model_type == ScoreModelType.Posterior:
            score_function_dict.update({"model": model})
        else:
            score_function_dict.update({"model": (model.likelihood_score, model.prior_score)})
    results = {k: [] for k in score_function_dict.keys()}
    for cond_loop in tqdm(in_problem.get_condition_list()):
        dataset = in_problem.get_dataset(number_of_samples, cond=cond_loop, iid_dataset=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results_per_bond = lbcrb_bound_per_cond(data_loader, score_function_dict, split_iid_in_eval=split_iid_in_eval)
        for k, v in results_per_bond.items():
            if len(v) == 1:
                results[k].append(v.get_fim().detach().cpu().numpy())
            else:
                results[k].append((v[0].get_fim().detach().cpu().numpy(), v[1].get_fim().detach().cpu().numpy()))
    result_function = {}
    for k, v in results.items():
        def _f(k_iid: int = 1):
            array_per_cond = np.asarray(v)
            if len(array_per_cond.shape) == 3:
                if k_iid != 1:
                    raise Exception("The array is not split iid")
                return np.linalg.inv(array_per_cond)
            elif len(array_per_cond.shape) == 4:
                if k_iid < 1:
                    raise ValueError("k_iid should be greater than 1")
                mfim = k_iid * array_per_cond[:, 0, :, :]
                pfim = array_per_cond[:, 1, :, :]

                return np.linalg.inv(mfim + pfim)

        result_function.update({k: _f})
    return result_function


def gbcrb_bound_per_cond(compute_optimal, data_loader, in_model, optimal_likelihood_score, optimal_prior_score,
                         score_model_type, split_iid_in_eval, remove_mean):
    norm_base_prior = []
    norm_base_lik = []
    norm_delta_prior = []
    norm_delta_lik = []
    efim_acc = Score2FIMAccumulator(split_iid_in_eval=split_iid_in_eval, remove_mean=remove_mean)
    epim_acc = Score2FIMAccumulator(split_iid_in_eval=split_iid_in_eval, remove_mean=remove_mean)
    epim_acc_opt = Score2FIMAccumulator(split_iid_in_eval=split_iid_in_eval)
    efim_acc_opt = Score2FIMAccumulator(split_iid_in_eval=split_iid_in_eval, rescale_iid=1) # TODO: Check this
    with torch.no_grad():

        for x, theta, cond in data_loader:
            x = x.to(pru.get_working_device())
            theta = theta.to(pru.get_working_device())
            cond = cond.to(pru.get_working_device())
            if score_model_type == ScoreModelType.Posterior:
                score = in_model(x, theta, cond)
            else:
                score = in_model.likelihood_score(x, theta, cond, sum_over_iid=False)
                prior_score = in_model.prior_score(theta)  # in_model.prior_score(theta)

            if compute_optimal:
                score_opt = optimal_likelihood_score(x, theta, cond)
                prior_score_opt = optimal_prior_score(theta)

                efim_acc_opt.accumulate(score_opt)
                epim_acc_opt.accumulate(prior_score_opt)

                norm_base_lik.append(torch.linalg.norm(score_opt, dim=-1))
                norm_base_prior.append(torch.linalg.norm(prior_score_opt, dim=-1))

                norm_delta_lik.append(torch.linalg.norm(score_opt - score.sum(dim=1), dim=-1))
                norm_delta_prior.append(torch.linalg.norm(prior_score_opt - prior_score, dim=-1))

            if score_model_type in [ScoreModelType.PriorLikelihoodSplitIID,
                                    ScoreModelType.PriorLikelihoodSplitIIDModelInformed]:
                efim_acc.accumulate(score)
            elif score_model_type == ScoreModelType.PriorLikelihoodSplit:

                efim_acc.accumulate(score)
            elif score_model_type == ScoreModelType.Posterior:

                efim_acc.accumulate(score)
            else:
                raise Exception("Unknown score model type")

            if score_model_type != ScoreModelType.Posterior:
                epim_acc.accumulate(prior_score)

    return (efim_acc.get_fim(), efim_acc.get_fim(), efim_acc_opt.get_fim(),
            norm_base_lik, norm_base_prior, norm_delta_lik, norm_delta_prior, epim_acc.get_fim(),
            epim_acc_opt.get_fim())


def compute_re_vs_param(in_run_list,
                        param_name="dataset_size",
                        run_est=False,
                        split_iid_in_eval=True,
                        number_of_samples=None,
                        k_iid_samples=16,
                        compute_optimal=False):
    """
    Compute the relative error vs the parameter
    :param in_run_list: The list of runs
    :param param_name: The parameter name
    :param run_est: Run the estimator
    :param split_iid_in_eval: Split the iid samples in the evaluation
    :param number_of_samples: Number of samples
    :param k_iid_samples: Number of iid samples
    :return: The relative error list, the parameter list, the problem snr array
    """
    re_list = []  # Add relative error list
    param_list = []  # Add dataset size
    gbcrb = []  # Add dataset size
    mmse_error = []
    for run_name in in_run_list:
        model, problem, param, _ = load_wandb_run(run_name)
        number_of_samples = param.dataset_size // problem.snr_array.shape[
            0] if number_of_samples is None else number_of_samples
        compute_bcrb, sigma_array = generate_bound_function(problem, model, param.score_model_type,
                                                            split_iid_in_eval=split_iid_in_eval,
                                                            number_of_samples=number_of_samples,
                                                            compute_optimal=compute_optimal)

        if param.score_model_type == ScoreModelType.PriorLikelihoodSplitIID:
            score_bcrb = np.stack([compute_bcrb(k_iid_samples, sigma) for sigma in sigma_array])
        else:
            score_bcrb = np.stack([compute_bcrb(sigma) for sigma in sigma_array])
        gbcrb.append(score_bcrb)

        if run_est:
            mmse, _ = train_mmse_estimator(problem)
            _mmse_array = []
            for cond in sigma_array:
                dataset = problem.get_dataset(number_of_samples, iid_dataset=True, cond=cond)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
                _mmse = 0
                for x, theta, snr in dataloader:
                    x = x.to(pru.get_working_device())
                    theta = theta.to(pru.get_working_device())
                    snr = snr.to(pru.get_working_device())
                    theta_hat = mmse(x, snr)
                    _mmse += torch.mean((theta - theta_hat) ** 2).item()
                _mmse /= len(dataloader)
                _mmse_array.append(_mmse)
            mmse_error.append(_mmse_array)
        param_list.append(getattr(param, param_name))

    return gbcrb, re_list, param_list, mmse_error, problem.get_condition_list()


def bcrb_relative_error(bcrb, score_bcrb):
    if bcrb.shape != score_bcrb.shape:
        raise ValueError("The shape of the BCRB and the score BCRB must match")
    if isinstance(bcrb, torch.Tensor):
        bcrb = bcrb.cpu().numpy()
    if isinstance(score_bcrb, torch.Tensor):
        score_bcrb = score_bcrb.cpu().numpy()
    if len(bcrb.shape) != 3:
        raise ValueError("The BCRB and the score BCRB must have a shape of (n, m, m)")
    return 100 * np.linalg.norm(bcrb - score_bcrb, axis=(1, 2), ord=2) / np.linalg.norm(bcrb, axis=(1, 2), ord=2)
