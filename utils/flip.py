import torch


def flip_lr(tensor, clone=False):
    """4-dimensional tensor, N, C, H, W"""
    w = tensor.size()[-1]
    newtensor = tensor[:, :, :, torch.arange(w - 1, -1, -1)]
    return newtensor.clone() if clone else newtensor
