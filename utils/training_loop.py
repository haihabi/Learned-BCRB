import numpy as np
import torch
import pyresearchutils as pru
import wandb

from tqdm import tqdm
from analysis.score2fim_accumulator import ConditionedScore2FIMAccumulator, Score2FIMAccumulator
from signal_models.base_problem import BaseProblem
from utils.config import ScoreModelType, OptimizerType
from utils.metric import ScoreResultAnalyzer
from utils.score_loss_context import prepare_input
from torch.utils import data


def log_lbcrb(in_ema, in_data_loader: data.DataLoader, in_problem: BaseProblem):
    print("Log LBCRB values")
    cs2f = ConditionedScore2FIMAccumulator(in_problem.get_condition_list())
    prior_fim = Score2FIMAccumulator()

    with torch.no_grad():
        for x, p, meta in in_data_loader:
            x = x.to(pru.get_working_device())
            p = p.to(pru.get_working_device())
            meta = meta.to(pru.get_working_device())
            p = prepare_input(p)
            with torch.no_grad():
                score = in_ema.module(x, p, meta)
                if len(score) == 2:
                    cs2f.accumulate(score[0], meta)
                    prior_fim.accumulate(score[1])
                else:
                    cs2f.accumulate(score, meta)
        if len(score) == 2:
            efim = cs2f.get_fim()
            pfim = prior_fim.get_fim()
            fim = efim + pfim
        else:
            fim = cs2f.get_fim()
        lbcrb = torch.diagonal(torch.linalg.inv(fim), dim1=-2, dim2=-1).sum(dim=-1)
        results = lbcrb.cpu().detach().numpy()
    for cond, r in zip(in_problem.get_condition_list(), results):
        wandb.log({"SNR": cond.item(),
                   "lbcrb": r})


def get_optimizer(in_model, in_param, param2update=None):
    if in_param.optimizer_type == OptimizerType.AdamW:
        return torch.optim.AdamW(in_model.parameters() if param2update is None else param2update,
                                 betas=(in_param.beta1, in_param.beta2),
                                 lr=in_param.lr,
                                 weight_decay=in_param.weight_decay, amsgrad=in_param.amsgrad)
    elif in_param.optimizer_type == OptimizerType.RMSprop:
        return torch.optim.RMSprop(in_model.parameters() if param2update is None else param2update, lr=in_param.lr,
                                   weight_decay=in_param.weight_decay)
    else:
        raise Exception("Unknown optimizer type")


def training_loop_joint(in_model, in_data_loader, in_loss_context, in_ema, in_val_loader, in_param,
                        in_problem: BaseProblem, param2update=None):
    opt = get_optimizer(in_model, in_param, param2update)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, div_factor=in_param.div_factor, max_lr=in_param.lr,
                                                       epochs=in_param.n_epochs, steps_per_epoch=len(in_data_loader),
                                                       pct_start=in_param.warmup_epochs / in_param.n_epochs)

    device = pru.get_working_device()
    ma = pru.MetricAveraging()
    print("Start Training")
    pbar = tqdm(range(in_param.n_epochs), desc='description')
    mod_factor = in_param.n_epochs // in_param.n_eval
    if mod_factor == 0:
        mod_factor = 1
    opt_prior_score_fn = in_problem.get_optimal_prior_score()
    opt_likelihood_score_fn = in_problem.get_optimal_likelihood_score()
    scr = ScoreResultAnalyzer(per_condition=True)
    score_analysis_data = {}
    for epoch in pbar:
        ma.clear()
        in_model.train()
        in_loss_context.train()

        for x, p, meta in in_data_loader:
            opt.zero_grad()
            x = x.to(device)
            p = p.to(device)
            meta = meta.to(device)
            p = prepare_input(p)
            loss, loss_dict, _ = in_loss_context.loss_function(in_model, x, p, meta)
            loss.backward()
            opt.step()
            in_ema.update_parameters(in_model)

            lr_scheduler.step()
            ma.log(**loss_dict)

        if (epoch % mod_factor == 0) or epoch == (in_param.n_epochs - 1):
            validate_score(device, in_ema, in_loss_context, in_model, in_param, in_val_loader, ma,
                           opt_likelihood_score_fn, opt_prior_score_fn, scr, in_problem)
            score_analysis_data = scr.analyze()
            if ma.is_best("loss_val_combined"):
                if score_analysis_data.get("score_analysis_data") is not None:
                    best_re = score_analysis_data["re_not_ema"]
                    best_re_ema = score_analysis_data["re_ema"]
                else:
                    best_re = best_re_ema = np.nan
                torch.save(in_model.state_dict(), "model_best.pt")
                torch.save(in_ema.state_dict(), "model_ema_best.pt")
                wandb.save("model_best.pt", policy="now")
                wandb.save("model_ema_best.pt", policy="now")
        torch.save(in_ema.state_dict(), "model_ema_last.pt")
        torch.save(in_model.state_dict(), "model_last.pt")
        wandb.save("model_last.pt", policy="now")
        wandb.save("model_ema_last.pt", policy="now")
        pbar.set_description(ma.results_str())
        data = ma.result
        data.update(score_analysis_data)
        data.update({"best_re": best_re, "best_re_ema": best_re_ema})
        wandb.log(data)
    log_lbcrb(in_ema, in_data_loader, in_problem)


class LearningQualityMetric:
    def __init__(self):
        self.data = []
        self.data_fim = []
        self.n_samples = []

    def accumulate(self, in_score, in_score_loss):
        C_approx = torch.mean(0.5 * torch.sum(in_score ** 2, dim=-1))
        l_tilde = torch.mean(in_score_loss)
        fim = torch.mean(in_score.unsqueeze(-1) @ in_score.unsqueeze(-2), dim=0)
        self.n_samples.append(in_score.shape[0])
        self.data.append(torch.stack([C_approx, l_tilde]).reshape([1, 2]))
        self.data_fim.append(fim)

    def result(self):
        data = torch.cat(self.data)
        data_fim = torch.stack(self.data_fim)
        w = torch.tensor(self.n_samples, device=data.device).float()
        w /= w.sum()

        c_approx, l_tilde = torch.sum(w.reshape([-1, 1]) * data, dim=0)
        fim = torch.linalg.norm(torch.sum(data_fim * w.reshape([-1, 1, 1]), dim=0), ord=2)
        l_approx = torch.abs(c_approx + l_tilde)
        metric = 100 * (l_approx / fim + 2 * torch.sqrt(l_approx / fim))

        return {"C_approx": c_approx.item(), "l_tilde": l_tilde.item(), "fim": fim.item(), "metric": metric.item()}


class ConditionalLearningQualityMetric:
    def __init__(self, condition_list):
        self.condition_list = condition_list
        self.results = {c.item(): LearningQualityMetric() for c in condition_list}

    def accumulate(self, in_score, in_score_loss, in_meta):
        for cond in self.condition_list:
            mask = in_meta == cond
            if torch.sum(mask) > 0:
                self.results[cond.item()].accumulate(in_score[mask], in_score_loss[mask])

    def get_metric(self) -> torch.Tensor:
        """
        Get the Fisher Information Matrix.
        :return: The Fisher Information Matrix.
        """
        metric_list = [self.results[condition.item()].result()
                       for condition in self.condition_list if self.results.get(condition.item()) is not None]
        return metric_list


def validate_score(device, in_ema, in_loss_context, in_model, in_param, in_val_loader, ma, opt_likelihood_score_fn,
                   opt_prior_score_fn, scr, in_problem):
    in_loss_context.eval()
    in_model.eval()
    scr.clear()

    cs2pfim = Score2FIMAccumulator()
    cs2efim = ConditionedScore2FIMAccumulator(in_problem.get_condition_list())
    cs2bfim = ConditionedScore2FIMAccumulator(in_problem.get_condition_list())
    lqm = ConditionalLearningQualityMetric(in_problem.get_condition_list())
    for x, p, meta in in_val_loader:
        x = x.to(device)
        p = p.to(device)
        meta = meta.to(device)
        p = prepare_input(p)

        loss_val, loss_dict_ema, data_ema = in_loss_context.loss_function(in_ema.module, x, p, meta,
                                                                          add_str="_ema")  # TODO: Correct head
        _, loss_dict_val, data = in_loss_context.loss_function(in_model, x, p, meta)  # TODO: Correct head
        if in_param.score_model_type.is_measurement_prior():
            lqm.accumulate(data_ema["likelihood_score"], data_ema["loss_likelihood_sm_per_sample"], meta)
        elif not in_param.score_model_type.is_measurement_prior():
            lqm.accumulate(data_ema["posterior_score"], data_ema["loss_posterior_sm_per_sample"], meta)
        else:
            raise Exception("Unknown score model type")

        if opt_likelihood_score_fn is not None:
            score_opt_lik = opt_likelihood_score_fn(x, p, meta)
            score_opt_prior = opt_prior_score_fn(p)
            with torch.no_grad():
                score_ema_prior_split = score_ema_split = score_not_ema_split = score_combined_ema = score_combined_not_ema = score_not_ema_prior_split = None
                if in_param.score_model_type in [ScoreModelType.PriorLikelihoodSplit,
                                                 ScoreModelType.PriorLikelihoodSplitIID,
                                                 ScoreModelType.PriorLikelihoodSplitIIDModelInformed]:
                    score_ema_split, score_ema_prior_split = data_ema["likelihood_score"], data_ema["prior_score"]
                    score_not_ema_split, score_not_ema_prior_split = data["likelihood_score"], data["prior_score"]

                    score_combined_ema = score_ema_split + score_ema_prior_split
                    score_combined_not_ema = score_not_ema_split + score_not_ema_prior_split
                elif in_param.score_model_type == ScoreModelType.Posterior:
                    score_combined_ema = data_ema["posterior_score"]
                    score_combined_not_ema = data["posterior_score"]
                else:
                    raise Exception("Unknown score model type")
                if score_opt_prior is not None and score_ema_prior_split is not None:
                    scr.add_result("ema_prior", score_ema_prior_split, score_opt_prior, meta)
                    scr.add_result("prior", score_not_ema_prior_split, score_opt_prior, meta)

                if score_opt_lik is not None and score_ema_split is not None:
                    scr.add_result("ema_lik", score_ema_split, score_opt_lik, meta)
                    scr.add_result("lik", score_not_ema_split, score_opt_lik, meta)

                if score_opt_prior is not None and score_combined_ema is not None and score_opt_lik is not None:
                    scr.add_result("ema", score_combined_ema, score_opt_lik + score_opt_prior, meta)
                    scr.add_result("not_ema", score_combined_not_ema, score_opt_lik + score_opt_prior, meta)
                if in_problem.has_bcrb:
                    if score_opt_prior is not None and score_ema_prior_split is not None:
                        cs2pfim.accumulate(score_ema_prior_split)
                    if score_opt_lik is not None and score_ema_split is not None:
                        cs2efim.accumulate(score_ema_split, meta)
                    if score_opt_prior is not None and score_combined_ema is not None:
                        cs2bfim.accumulate(score_opt_prior + score_opt_lik, meta)
        if in_problem.has_bcrb:
            referece_per_cond = in_problem.bcrb_vs_condition()
            if score_opt_prior is not None and score_ema_prior_split is not None:
                pfim = cs2pfim.get_fim()
                ref_pfim = referece_per_cond[0, -1, :, :]
                pfim_re, pfim_nd = compute_fim_error(pfim, ref_pfim, return_norm=True)
                loss_dict_ema.update({"pfim_re": pfim_re, "pfim_nd": pfim_nd})
            if score_opt_lik is not None and score_ema_split is not None:
                efim = cs2efim.get_fim()
                ref_efim = referece_per_cond[:, 1, :, :]
                efim_re, efim_nd = compute_fim_error(efim, ref_efim, return_norm=True, mean=False)
                loss_dict_ema.update(
                    {f"cond_{str(c.item())}/efim_nd": efim_nd[i].item() for i, c in enumerate(cs2efim.condition_list)})
                loss_dict_ema.update(
                    {f"cond_{str(c.item())}/efim_re": efim_re[i].item() for i, c in enumerate(cs2efim.condition_list)})

        ma.log(**loss_dict_ema,
               **loss_dict_val,
               loss_val_combined=loss_val.item())
    lqm_result = lqm.get_metric()
    results_dict = {}
    for c, r in zip(in_problem.get_condition_list(), lqm_result):
        results_dict.update({f"cond_{str(c.item())}/C_approx": r["C_approx"],
                             f"cond_{str(c.item())}/l_tilde": r["l_tilde"],
                             f"cond_{str(c.item())}/metric": r["metric"],
                             f"cond_{str(c.item())}/fim_norm": r["fim"]})
    ma.log(**results_dict)


def compute_fim_error(in_fim: torch.Tensor, in_ref_fim: torch.Tensor, mean=True, return_norm=False, ord=2):
    if in_fim.shape != in_ref_fim.shape:
        raise Exception("The shapes of the input and reference FIMs do not match")

    if len(in_fim.shape) not in [2, 3]:
        raise Exception("The input FIM has an unexpected shape")
    if len(in_fim.shape) == 3:
        dims = (1, 2)
    else:
        dims = (0, 1)
    if in_fim.shape[dims[0]] != in_fim.shape[dims[1]]:
        raise Exception("The input FIM is not square")
    norm_diff = torch.linalg.norm(in_ref_fim - in_fim, ord=ord, dim=dims)
    norm = torch.linalg.norm(in_ref_fim, ord=ord, dim=dims)
    if return_norm:
        if mean:
            return 100 * (torch.mean(norm_diff / norm)).item(), torch.mean(norm).item()
        else:
            return 100 * (norm_diff / norm).cpu().detach().numpy(), norm.cpu().detach().numpy()
    else:
        return 100 * torch.mean(norm_diff / norm).item()


def compute_score_re(in_score, in_ref_score):
    if in_score.shape != in_ref_score.shape:
        raise Exception("The shapes of the input and reference scores do not match")
    return 100 * (torch.mean(torch.linalg.norm(in_ref_score - in_score, dim=-1)) / torch.mean(
        torch.linalg.norm(in_ref_score, dim=-1))).item()
