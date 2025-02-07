from argparse import Namespace

import torch
import wandb

import pyresearchutils as pru
from neural_networks.basic_blocks.non_linearity_factory import NonLinearityType
from utils.constants import PROJECT, MODEL_PROBLEM_PATH
from utils.builder import param2problem, param2score_nn
from utils.config import get_default_config


def init_run(is_sweep, is_eval=False):
    cr = get_default_config()
    param, log_dir = pru.initialized_log(PROJECT, cr, enable_wandb=not is_eval)  # Initialize the log
    param_dict = vars(param)
    if not is_eval:
        for key in ["lr", "n_layers", "feature_size", "n_epochs", "batch_size", "weight_decay", "non_linearity"]:
            if key == "non_linearity":
                if is_sweep:
                    param_dict[key] = getattr(NonLinearityType, wandb.config[key])
            else:
                param_dict[key] = wandb.config[key]
    param = Namespace(**param_dict)
    device = pru.get_working_device()  # Get the device
    problem = param2problem(param)
    data_loader, val_loader = create_dataloaders(param, problem)

    if is_eval:
        problem.load_model(MODEL_PROBLEM_PATH)  # Load the model
    else:
        problem.save_model(MODEL_PROBLEM_PATH)  # Save the model
        wandb.save(MODEL_PROBLEM_PATH, policy="now")  # Upload the model
    model = param2score_nn(device, param, problem)
    return data_loader, device, model, param, problem, val_loader


def create_dataloaders(param, problem):
    dataset = problem.get_dataset(param.dataset_size + int(0.2 * param.dataset_size),
                                  iid_dataset=True)  # Get the training dataset
    training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [param.dataset_size,
                                                                                   int(0.2 * param.dataset_size)])
    data_loader = torch.utils.data.DataLoader(training_dataset, param.batch_size, shuffle=True,
                                              num_workers=1, pin_memory=True,
                                              persistent_workers=True)  # Create the data loader
    val_loader = torch.utils.data.DataLoader(validation_dataset, param.batch_size,
                                             num_workers=1, pin_memory=True, shuffle=False,
                                             persistent_workers=True)  # Create the validation data loader
    return data_loader, val_loader
