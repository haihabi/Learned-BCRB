import os
import sys

cwd = os.getcwd()
print(cwd)
sys.path.append(os.path.join(os.path.split(cwd)[0], "py-research-utils"))

from utils.init_run import init_run
from score import score_matching_loss
from score.ema import ModelEma
from utils.config import ScoreModelType
from utils.prior_training import training_loop_prior
from utils.score_loss_context import TrainingPart, ScoreLossContext
from utils.training_loop import training_loop_joint
import torch
import wandb


def run_main(is_sweep=False):
    data_loader, device, model, param, problem, val_loader = init_run(is_sweep)

    if param.score_model_type != ScoreModelType.Posterior:
        if param.optimal_prior:
            print("Use Optimal Prior")
            model.set_optimal_prior(problem.get_optimal_prior_score())
        else:
            loss_context = score_matching_loss.ScoreMatchingLoss()

            ema_prior = ModelEma(model.mlp_prior, decay=param.ema_decay)
            best_ema = training_loop_prior(model.mlp_prior, data_loader, loss_context, ema_prior, val_loader, param,
                                           problem.get_optimal_prior_score(), problem.get_prior_fim())

            model.mlp_prior.load_state_dict(best_ema.state_dict())
        run_nn_score_training(data_loader, device, model, param, problem, val_loader, TrainingPart.Likelihood,
                              param2update=model.mlp.parameters())

    else:
        run_nn_score_training(data_loader, device, model, param, problem, val_loader, TrainingPart.Posterior,
                              param2update=None)
    wandb.finish()


def run_nn_score_training(data_loader, device, model, param, problem, val_loader, training_part, param2update):
    ema = torch.optim.swa_utils.AveragedModel(model,
                                              multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                                                  param.ema_decay))
    loss_context = ScoreLossContext(-torch.inf * torch.ones(1).to(device),
                                    torch.inf * torch.ones(1).to(device),
                                    param.loss_type,
                                    training_part=training_part,
                                    score_model_type=param.score_model_type,
                                    in_condition_list=problem.get_condition_list())
    training_loop_joint(model, data_loader, loss_context, ema, val_loader, param, problem, param2update=param2update)


if __name__ == '__main__':
    if False:
        sweep_configuration = {
            "method": "bayes",
            "metric": {"goal": "minimize", "name": "loss_likelihood_sm_val"},
            "parameters": {
                "lr": {"min": 0.00001, "max": 0.001},
                "n_layers": {"values": [2, 3, 4, 5]},
                "batch_size": {"values": [512]},
                "n_epochs": {"values": [300]},
                "feature_size": {"values": [32, 48, 64, 96, 128, 256, 512, 1024]},
                "weight_decay": {"values": [0.0, 1e-1, 1e-2, 1e-3, 1e-4]},
                "non_linearity": {"values": NON_LINEARITY_LIST},
            },
        }


        def run_sweep():
            run_main(True)


        sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT)

        wandb.agent(sweep_id, function=run_sweep, count=50)
    else:
        run_main()
