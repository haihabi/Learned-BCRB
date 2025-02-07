import torch


def load_ema_state_dict(device, file_name, model):
    weight_dict = torch.load(file_name, map_location=device, weights_only=True)
    if "ema" in file_name:
        weight_dict = {".".join(k.split(".")[1:]) if "module" in k else k: v for k, v in weight_dict.items()}
        if weight_dict.get("n_averaged") is not None:
            weight_dict.pop("n_averaged")
    model.load_state_dict(weight_dict)
