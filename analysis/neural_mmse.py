import copy

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from neural_networks.basic_blocks.adpative_normalization import SelectiveAdaptiveNormalization
from neural_networks.basic_blocks.mlp import MLP
from pyresearchutils.initlized_log import wandb
from signal_models.base_problem import BaseProblem
import pyresearchutils as pru
import matplotlib.pyplot as plt


def db(x):
    return 10 * np.log10(x)


class NMMSEResult:
    def __init__(self, conditions):
        self.conditions = conditions
        self.acc_dict = {cond.item(): [] for cond in conditions}
        self.count_dict = {cond.item(): [] for cond in conditions}

    def accumulate(self, mse, condition):
        for cond in self.conditions:
            index = condition == cond
            if torch.sum(index) == 0:
                continue
            self.acc_dict[cond.item()].append(torch.sum(mse[index]))
            self.count_dict[cond.item()].append(torch.sum(index))

    def get_results(self):
        results = []
        for cond in self.conditions:
            mse = torch.stack(self.acc_dict[cond.item()]).sum().item()
            count = torch.stack(self.count_dict[cond.item()]).sum().item()
            results.append([cond.item(), mse / count, count])
        return np.asarray(results)


def run_validation_loop(dataloader_val, mmse_estimator, device):
    with torch.no_grad():
        mmse_estimator.eval()
        loss_avg_val = 0
        for x, theta, snr in dataloader_val:
            x = x.to(device)
            theta = theta.to(device)
            snr = snr.to(device)
            theta_hat = mmse_estimator(x, snr)
            loss = torch.mean((theta - theta_hat) ** 2)
            loss_avg_val += loss.item()
        loss_avg_val /= len(dataloader_val)
    return loss_avg_val


class MMSEEstimatorModel(nn.Module):
    def __init__(self, estimation_function, n, m, k, n_layers, feature_size, ondition_list, selective=True,
                 in_meta_size=1,
                 non_linearity=nn.SiLU, droupout=0.0):
        super().__init__()
        self.delta_estimator = MMSEEstimator(n + m, m, k, n_layers, feature_size, ondition_list, selective=True,
                                             in_meta_size=in_meta_size,
                                             non_linearity=non_linearity, droupout=droupout)
        self.estimation_function = estimation_function
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, snr):
        theta_hat = self.estimation_function(x)
        delta = self.delta_estimator(torch.cat([x, theta_hat.unsqueeze(-1)], dim=-1), snr)
        return theta_hat + delta * self.scale


def train_mmse_estimator(in_problem: BaseProblem, m_epochs=200, lr=1e-3, debug=True,
                         train_size=200000, val_size=10000, estimation_function=None, pct_start=0.1, div_factor=10,
                         weight_decay=1e-2,
                         condition=None):
    """
    Train MMSE estimator

    :param in_problem:
    :param m_epochs:
    :param lr:
    :param debug:
    :param train_size:
    :param val_size:
    :return:
    """
    if debug:
        wandb.init(project='mmse_estimator')
        wandb.config.update(
            {'m_epochs': m_epochs, 'lr': lr, 'train_size': train_size, 'val_size': val_size, 'pct_start': pct_start,
             'div_factor': div_factor, "weight_decay": weight_decay,
             'condition': condition})
    device = pru.get_working_device()
    dataset = in_problem.get_dataset(train_size + val_size, iid_dataset=True, cond=condition)
    training_data, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    dataloader = torch.utils.data.DataLoader(training_data, batch_size=512, shuffle=True)

    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False)
    if estimation_function is None:
        mmse_estimator = MMSEEstimator(in_problem.n, in_problem.m, in_problem.k, 3, 128,
                                       in_problem.get_condition_list(),
                                       droupout=0.0).to(
            device)
    else:
        mmse_estimator = MMSEEstimatorModel(estimation_function, in_problem.n, in_problem.m, in_problem.k, 3, 128,
                                            in_problem.get_condition_list(), droupout=0.0).to(
            device)
    optimizer = torch.optim.AdamW(mmse_estimator.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)

    pbar = tqdm(range(m_epochs), desc='description')
    loss_avg_val_best = 1e8
    results_list = []

    for epoch in range(m_epochs):

        mmse_estimator.train()
        loss_avg = 0
        for x, theta, snr in dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            theta = theta.to(device)
            snr = snr.to(device)
            theta_hat = mmse_estimator(x, snr)
            loss = torch.mean((theta - theta_hat) ** 2)
            loss.backward()
            loss_avg += loss.item()
            optimizer.step()
        # lr_scheduler.step()
        loss_avg /= len(dataloader)
        loss_avg_val = run_validation_loop(dataloader_val, mmse_estimator, device)
        if loss_avg_val < loss_avg_val_best:
            loss_avg_val_best = loss_avg_val
            mmse_estimator_best = copy.deepcopy(mmse_estimator)
        results_list.append([loss_avg, loss_avg_val, loss_avg_val_best])
        pbar.update(1)
        if debug:
            wandb.log({'loss': loss_avg, 'val_loss': loss_avg_val, 'best_val_loss': loss_avg_val_best})
        pbar.set_description(
            f"Epoch: {epoch}, Loss: {loss_avg}, Val Loss: {loss_avg_val}, Best Val Loss: {loss_avg_val_best}")
    if debug:
        results_list = np.array(results_list)

        plt.semilogy(results_list[:, 0], label='Train')
        plt.semilogy(results_list[:, 1], "--", label='Val')
        plt.semilogy(results_list[:, 2], label='Best Val')
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.legend()
        plt.show()
    mmse_estimator_best.eval()

    nmmse_results = NMMSEResult(in_problem.get_condition_list() if condition is None else [condition])
    for x, theta, snr in dataloader_val:
        x = x.to(pru.get_working_device())
        theta = theta.to(pru.get_working_device())
        snr = snr.to(pru.get_working_device())
        theta_hat = mmse_estimator_best(x, snr)
        nmmse_results.accumulate(torch.mean((theta - theta_hat) ** 2, dim=-1), snr)

    return mmse_estimator_best, nmmse_results.get_results()

    # pass


class MMSEEstimator(torch.nn.Module):
    def __init__(self, n, m, k, n_layers, feature_size, snr_list, selective=True, in_meta_size=1,
                 non_linearity=nn.SiLU, droupout=0.0):
        super().__init__()
        self.n = n
        self.m = m
        self.k = k

        def normalization():
            return SelectiveAdaptiveNormalization(in_meta_size, snr_list)

        self.mlp = MLP(n * k, m, feature_size, n_layers, non_linearity=non_linearity,
                       normalization=normalization,
                       bias_output=True, se_block=True, droupout=droupout)

    def forward(self, x, snr):
        x = x.reshape([x.shape[0], -1])
        return self.mlp(x, snr)
