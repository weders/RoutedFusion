import torch


def rmse(est, target, mask=None):

    if mask:
        metric = torch.sqrt(torch.sum(mask * torch.pow(est - target, 2)) / torch.sum(mask))
    else:
        metric = torch.sqrt(torch.mean(torch.pow(est - target, 2)))

    return metric


def mse(est, target, mask=None):

    if mask:
        metric = torch.sum(mask * torch.pow(est - target, 2)) / torch.sum(mask)
    else:
        metric = torch.mean(torch.pow(est - target, 2))

    return metric


def mad(est, target, mask=None):

    if mask:
        metric = torch.sum(mask * torch.abs(est - target)) / torch.sum(mask)
    else:
        metric = torch.mean(torch.abs(est - target))

    return metric

