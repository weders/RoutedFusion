import torch

import numpy as np


class FusionLoss(torch.nn.Module):

    def __init__(self, config, reduction='none', l1=True, l2=True, cos=True):
        super(FusionLoss, self).__init__()

        self.criterion1 = torch.nn.L1Loss(reduction=reduction)
        self.criterion2 = torch.nn.MSELoss(reduction=reduction)
        self.criterion3 = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction=reduction)

        self.lambda1 = 1. if l1 else 0.
        self.lambda2 = 0. if l2 else 0.
        self.lambda3 = 0.1   if cos else 0.

    def forward(self, est, target):

        if est.shape[1] == 0:
           return torch.ones_like(est).sum().clamp(min=1)

        x1 = torch.sign(est)
        x2 = torch.sign(target)

        x1 = x1[:, :, :]
        x2 = x2[:, :, :]

        label = torch.ones_like(x1)

        #TODO: clamping
        # est = torch.clamp(est, -0.1, 0.1)
        # target = torch.where(torch.abs(target) >= 0.01, 0.01*torch.ones_like(target), target)

        l1 = self.criterion1.forward(est, target)
        l2 = self.criterion2.forward(est, target)
        l3 = self.criterion3.forward(x1, x2, label)

        normalization = torch.ones_like(l1).sum()

        l_vis = l1 + l2 + l3
        l_vis /= normalization

        l1 = l1.sum() / normalization
        l2 = l2.sum() / normalization
        l3 = l3.sum() / normalization

        l = self.lambda1*l1 + self.lambda2*l2 + self.lambda3*l3

        return l


class RoutingLoss(torch.nn.Module):

    def __init__(self, config):

        super(RoutingLoss, self).__init__()

        self.criterion1 = GradientWeightedDepthLoss(config)
        self.criterion2 = UncertaintyDepthLoss(config)

    def forward(self, prediction, uncertainty, target, gradient_mask=None):

        l1 = self.criterion1.forward(prediction, target, gradient_mask)
        l2 = self.criterion2.forward(prediction, uncertainty, target, gradient_mask)
        return l1 + l2


class GradientWeightedDepthLoss(torch.nn.Module):
    """
    A simple L1 loss, but restricted to the cropped center of the image.
    It also does not count pixels outside of a given range of values (in target).
    Additionally, there is also an L1 loss on the gradient.
    """
    def __init__(self, config, crop_fraction=0.0, vmin=0, vmax=1, limit=10, weight_scale=1.0):
        """
        The input should be (batch x channels x height x width).
        We L1-penalize the inner portion of the image,
        with crop_fraction cut off from all borders.
        Keyword arguments:
            crop_fraction -- fraction to cut off from all sides (defaults to 0.25)
            vmin -- minimal (GT!) value to supervise
            vmax -- maximal (GT!) value to supervise
            limit -- anything higher than this is wrong, and should be ignored
        """
        super(GradientWeightedDepthLoss, self).__init__()

        self.weight_scale = config.weight_scale
        self.limit = config.limit
        self.crop_fraction = config.crop_fraction
        "Cut-off fraction"

        self.vmin = config.vmin
        "Lower bound for valid target pixels"

        self.vmax = config.vmax
        "Upper bound for valid target pixels"

        self.sobel_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_x.weight = torch.nn.Parameter(torch.from_numpy(np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])/8.).float().unsqueeze(0).unsqueeze(0))
        self.sobel_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y.weight = torch.nn.Parameter(torch.from_numpy(np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])/8.).float().unsqueeze(0).unsqueeze(0))

        gpu = torch.device('cuda')
        self.sobel_x = self.sobel_x.to(gpu)
        self.sobel_y = self.sobel_y.to(gpu)

    def forward(self, input, target, gradient_mask=None):
        height = input.size(2)
        heightcrop = int(height * self.crop_fraction)
        width = input.size(3)
        widthcrop = int(width * self.crop_fraction)

        if self.crop_fraction > 0:
            input_crop = input[:,:,heightcrop:height-heightcrop,widthcrop:width-widthcrop]
            target_crop = target[:,:,heightcrop:height-heightcrop,widthcrop:width-widthcrop]
        else:
            input_crop = input
            target_crop = target

        valid_mask = (target_crop.le(self.vmax) * target_crop.ge(self.vmin)).float()

        input_gradx = self.sobel_x(input_crop)
        input_grady = self.sobel_y(input_crop)

        target_gradx = self.sobel_x(target_crop)
        target_grady = self.sobel_y(target_crop)

        grad_maskx = self.sobel_x(valid_mask)
        grad_masky = self.sobel_y(valid_mask)
        grad_valid_mask = (grad_maskx.eq(0) * grad_masky.eq(0)).float()*valid_mask

        if gradient_mask is not None:
            grad_valid_mask[gradient_mask == 0] = 0

        gradloss = torch.abs( (input_gradx - target_gradx) ) + torch.abs( (input_grady - target_grady) )

        # weight l1 loss with gradient
        weights = self.weight_scale*gradloss + torch.ones_like(gradloss)
        gradloss = (gradloss * grad_valid_mask ).sum()
        gradloss = gradloss / grad_valid_mask.sum().clamp(min=1)

        loss = torch.abs((input_crop - target_crop) * valid_mask)
        loss = torch.mul(weights, loss).sum()
        loss = loss / valid_mask.sum().clamp(min=1)

        loss = loss + gradloss

        # if this loss value is not plausible, cap it (which will also not backprop gradients)
        if self.limit is not None and loss > self.limit:
            loss = torch.clamp(loss, max=self.limit)

        if loss.ne(loss).item():
            print("Nan loss!")

        return loss

class UncertaintyDepthLoss(torch.nn.Module):
    """
    A simple L1 loss, but restricted to the cropped center of the image.
    It also does not count pixels outside of a given range of values (in target).
    Additionally, there is also an L1 loss on the gradient.
    """
    def __init__(self, config):
        """
        The input should be (batch x channels x height x width).
        We L1-penalize the inner portion of the image,
        with crop_fraction cut off from all borders.
        Keyword arguments:
            crop_fraction -- fraction to cut off from all sides (defaults to 0.25)
            vmin -- minimal (GT!) value to supervise
            vmax -- maximal (GT!) value to supervise
            limit -- anything higher than this is wrong, and should be ignored
        """
        super().__init__()

        self.lambda_unc = config.lambda_unc

        self.crop_fraction = config.crop_fraction
        "Cut-off fraction"

        self.vmin = config.vmin
        "Lower bound for valid target pixels"

        self.vmax = config.vmax
        "Upper bound for valid target pixels"

        self.sobel_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_x.weight = torch.nn.Parameter(torch.from_numpy(np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])/8.).float().unsqueeze(0).unsqueeze(0))
        self.sobel_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y.weight = torch.nn.Parameter(torch.from_numpy(np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])/8.).float().unsqueeze(0).unsqueeze(0))

        gpu = torch.device('cuda')
        self.sobel_x = self.sobel_x.to(gpu)
        self.sobel_y = self.sobel_y.to(gpu)

        self.limit = config.limit

    def forward(self, input, uncertainty, target, gradient_mask=None):
        height = input.size(2)
        heightcrop = int(height * self.crop_fraction)
        width = input.size(3)
        widthcrop = int(width * self.crop_fraction)

        if self.crop_fraction > 0:
            input_crop = input[:,:,heightcrop:height-heightcrop,widthcrop:width-widthcrop]
            target_crop = target[:,:,heightcrop:height-heightcrop,widthcrop:width-widthcrop]
        else:
            input_crop = input
            target_crop = target

        valid_mask = (target_crop.le(self.vmax) * target_crop.ge(self.vmin)).float()
        valid_mask[target == 0] = 0

        input_gradx = self.sobel_x(input_crop)
        input_grady = self.sobel_y(input_crop)

        target_gradx = self.sobel_x(target_crop)
        target_grady = self.sobel_y(target_crop)

        grad_maskx = self.sobel_x(valid_mask)
        grad_masky = self.sobel_y(valid_mask)
        grad_valid_mask = (grad_maskx.eq(0) * grad_masky.eq(0)).float()*valid_mask
        grad_valid_mask[target == 0] = 0

        if gradient_mask is not None:
            grad_valid_mask[gradient_mask == 0] = 0

        s_i = uncertainty
        p_i = torch.exp(-1. * s_i)

        gradloss = torch.abs( (input_gradx - target_gradx) ) + torch.abs( (input_grady - target_grady) )
        gradloss = (gradloss * grad_valid_mask )
        gradloss = torch.mul(p_i, gradloss).sum()
        gradloss = gradloss / grad_valid_mask.sum().clamp(min=1)

        loss = torch.abs((input_crop - target_crop) * valid_mask)
        loss = torch.mul(loss, p_i).sum()
        loss = loss / valid_mask.sum().clamp(min=1)

        # sum of loss terms with uncertainty included
        loss = loss + gradloss + self.lambda_unc*0.5*uncertainty.sum()/valid_mask.sum().clamp(min=1)

        # if this loss value is not plausible, cap it (which will also not backprop gradients)
        if self.limit is not None and loss > self.limit:
            loss = torch.clamp(loss, max=self.limit)

        if loss.ne(loss).item():
            print("Nan loss!")

        return loss
