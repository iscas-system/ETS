import torch


# todo add to pyprof
def get_total_params(model: torch.nn.Module):
    sum(p.numel() for p in model.parameters())


def get_total_params_size(model: torch.nn.Module):
    sum(p.numel() * p.element_size() for p in model.parameters())


def get_total_params_trainable(model: torch.nn.Module):
    sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_params_size_trainable(model: torch.nn.Module):
    sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
