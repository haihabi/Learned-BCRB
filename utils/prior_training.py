import copy

import torch
import wandb
from tqdm import tqdm

import pyresearchutils as pru
from utils.metric import ScoreResultAnalyzer
from utils.score_loss_context import prepare_input


def training_loop_prior(in_model, in_data_loader, in_loss_context, in_ema, in_val_loader, in_param,
                        optimal_score_function=None,
                        optimal_prior_fim=None):
    opt = torch.optim.Adam(in_model.parameters(), lr=in_param.lr_prior, weight_decay=in_param.weight_decay_prior,
                           amsgrad=in_param.amsgrad)
    best_ema = None
    device = pru.get_working_device()
    ma = pru.MetricAveraging()
    print("Start Training")
    pbar = tqdm(range(in_param.n_epochs_prior), desc='description')
    mod_factor = max(in_param.n_epochs_prior // in_param.n_eval, 1)
    results = {}
    for epoch in pbar:
        ma.clear()
        if epoch % mod_factor == 0:
            in_loss_context.eval()
            in_model.eval()
            results = validation_sample_loop(device, in_ema, in_loss_context, in_model, in_val_loader, ma,
                                             optimal_score_function,
                                             optimal_prior_fim)

        in_model.train()
        in_loss_context.train()
        training_sample_loop_prior(device, in_data_loader, in_ema, in_loss_context, in_model, ma, opt)

        if ma.is_best("loss_val_prior_ema"):
            best_ema = copy.deepcopy(in_ema)

        pbar.set_description(ma.results_str())
        data = ma.result
        data.update(results)
        wandb.log(data)

    return best_ema


def validation_sample_loop(device, in_ema, in_loss_context, in_model, in_val_loader, ma, optimal_score_function,
                           optimal_prior_fim):
    scr = ScoreResultAnalyzer()
    est_pfim_ema = 0
    est_pfim_not_ema = 0
    for x, p, meta in in_val_loader:
        p = p.to(device)
        p = prepare_input(p)

        score_ema = in_ema(p)
        score_not_ema = in_model(p)
        if optimal_score_function is not None:
            score_opt = optimal_score_function(p)
            scr.add_result("prior_ema", score_ema, score_opt, meta)
            scr.add_result("prior", score_not_ema, score_opt, meta)
        if optimal_prior_fim is not None:
            est_pfim_not_ema += torch.mean(score_not_ema.unsqueeze(dim=-1) @ score_not_ema.unsqueeze(dim=-2), dim=0)
            est_pfim_ema += torch.mean(score_ema.unsqueeze(dim=-1) @ score_ema.unsqueeze(dim=-2), dim=0)

        loss_ema = in_loss_context(score_ema, p)
        loss_val = in_loss_context(score_not_ema, p)

        ma.log(loss_val_prior_ema=loss_ema.item(), loss_val_prior=loss_val.item())
    if optimal_prior_fim is not None:
        est_pfim_not_ema /= len(in_val_loader)
        est_pfim_ema /= len(in_val_loader)
        delta_fim = torch.linalg.norm(est_pfim_ema - optimal_prior_fim, ord=2)
        delta_fim_not_ema = torch.linalg.norm(est_pfim_not_ema - optimal_prior_fim, ord=2)
        norm_prior_fim = torch.linalg.norm(optimal_prior_fim, ord=2)
        ma.log(delta_fim=delta_fim.item(), delta_fim_not_ema=delta_fim_not_ema.item(),
               norm_prior_fim=norm_prior_fim.item())
    return scr.analyze()


def training_sample_loop_prior(device, in_data_loader, in_ema, in_loss_context, in_model, ma, opt):
    for x, p, meta in in_data_loader:
        opt.zero_grad()
        p = p.to(device)
        p = prepare_input(p)
        score_out = in_model(p)
        loss = in_loss_context(score_out, p)
        loss.backward()
        opt.step()
        in_ema.update(in_model)
        ma.log(loss_prior=loss.item())
