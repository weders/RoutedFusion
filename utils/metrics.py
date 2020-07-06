import numpy as np
import torch


def evaluation(est, target, mask=None, value=0.1):

    est = np.clip(est, -value, value)
    target = np.clip(target, -value, value)

    mse = mse_fn(est, target, mask)
    mad = mad_fn(est, target, mask)
    iou = iou_fn(est, target, mask)
    acc = acc_fn(est, target, mask)

    return {'mse': mse,
            'mad': mad,
            'iou': iou,
            'acc': acc}


def rmse_fn(est, target, mask=None):

    if mask is not None:
        metric = np.sqrt(np.sum(mask * np.power(est - target, 2)) / np.sum(mask))
    else:
        metric = np.sqrt(np.mean(np.power(est - target, 2)))

    return metric


def mse_fn(est, target, mask=None):

    if mask is not None:
        metric = np.sum(mask * np.power(est - target, 2)) / np.sum(mask)
    else:
        metric = np.mean(np.power(est - target, 2))

    return metric


def mad_fn(est, target, mask=None):

    if mask is not None:
        metric = np.sum(mask * np.abs(est - target)) / np.sum(mask)
    else:
        metric = np.mean(np.abs(est - target))

    return metric


def iou_fn(est, target, mask=None):

    if mask is not None:
        tp = (est < 0) & (target < 0) & (mask > 0)
        fp = (est < 0) & (target >= 0) & (mask > 0)
        fn = (est >= 0) & (target < 0) & (mask > 0)
    else:
        tp = (est < 0) & (target < 0)
        fp = (est < 0) & (target >= 0)
        fn = (est >= 0) & (target < 0)

    intersection = tp.sum()
    union = tp.sum() + fp.sum() + fn.sum()

    metric = intersection / union
    return metric


def acc_fn(est, target, mask=None):

    if mask is not None:
        tp = (est < 0) & (target < 0) & (mask > 0)
        tn = (est >= 0) & (target >= 0) & (mask > 0)
    else:
        tp = (est < 0) & (target < 0)
        tn = (est >= 0) & (target >= 0)

    acc = (tp.sum() + tn.sum()) / mask.sum()
    metric = acc
    return metric


def l2(est, gt, mask=None):

    # mask out tsdf where est is uninitialized
    if mask is None:
        torch.ones_like(est)

    l2 = torch.pow(est - gt, 2)
    l2 = l2.sum() / mask.sum()

    return l2.item()


def l1(est, gt, mask=None):

    if mask is None:
        est = est.ge
    loss = torch.abs(est - gt)
    loss = loss.sum() / mask.sum()

    return loss.item()


def accuracy(est, gt, mode='freespace'):

    mask_gt = torch.where(torch.abs(gt) < 0.05, torch.ones_like(gt), torch.zeros_like(gt)).int()
    mask_est = torch.where(torch.abs(est) < 0.05, torch.ones_like(est), torch.zeros_like(est)).int()

    mask = mask_gt | mask_est
    mask = mask.float()
    est = mask * est
    gt = mask * gt

    if mode == 'freespace':
        est_p = torch.where(est > 0, torch.ones_like(est), torch.zeros_like(est)).int()
        gt_p = torch.where(gt > 0, torch.ones_like(gt), torch.zeros_like(gt)).int()
        est_n = torch.where(est < 0, torch.ones_like(est), torch.zeros_like(est)).int()
        gt_n = torch.where(gt < 0, torch.ones_like(gt), torch.zeros_like(gt)).int()

    elif mode == 'occupied':
        est_p = torch.where(est < 0, torch.ones_like(est), torch.zeros_like(est)).int()
        gt_p = torch.where(gt < 0, torch.ones_like(gt), torch.zeros_like(gt)).int()
        est_n = torch.where(est > 0, torch.ones_like(est), torch.zeros_like(est)).int()
        gt_n = torch.where(gt > 0, torch.ones_like(gt), torch.zeros_like(gt)).int()


    tp = torch.sum(est_p & gt_p).item()
    fp = torch.sum(est_p).item() - tp
    tn = torch.sum(est_n & gt_n).item()
    fn = torch.sum(est_n).item() - tn

    sum = tp + fp + fn + tn
    if sum == 0.:
        sum = 1.0
        print('invalid volume')

    accuracy = (tp + tn)/sum

    return 100. * accuracy


def intersection_over_union(est, gt, mode='occupied', mask=None):

    if mask is None:
       mask = torch.ones_like(est)

    est = est * mask
    gt = gt * mask

    # compute occupancy
    if mode == 'occupied':
        est = est < 0.
        gt = gt < 0.
    else:
        est = est > 0.
        gt = gt > 0.

    union = torch.sum((est + gt) > 0.).item()

    if union == 0.:
        union = 1.

    intersection = torch.sum(est & gt).item()

    return intersection/union


def mean_absolute_distance(est, gt, mask=None):
    if mask is None:
        mask = torch.ones_like(est)
    absolute_distance = torch.abs(est - gt)
    return (torch.sum(absolute_distance) / torch.sum(mask)).item()
