import torch

import numpy as np


class RoutingLoss(torch.nn.Module):

    def __init__(self, config):

        super(RoutingLoss, self).__init__()


        self.criterion1 = GradientWeightedDepthLoss(crop_fraction=config.LOSS.crop_fraction,
                                                    vmin=config.LOSS.vmin,
                                                    vmax=config.LOSS.vmax,
                                                    weight_scale=config.LOSS.weight_scale)

        self.criterion2 = UncertaintyDepthLoss(crop_fraction=config.LOSS.crop_fraction,
                                               vmin=config.LOSS.vmin,
                                               vmax=config.LOSS.vmax)

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
    def __init__(self, crop_fraction=0.0, vmin=0, vmax=1, limit=10, weight_scale=1.0):
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

        self.weight_scale = weight_scale

        self.crop_fraction = crop_fraction
        "Cut-off fraction"

        self.vmin = vmin
        "Lower bound for valid target pixels"

        self.vmax = vmax
        "Upper bound for valid target pixels"

        self.sobel_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_x.weight = torch.nn.Parameter(torch.from_numpy(np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])/8.).float().unsqueeze(0).unsqueeze(0))
        self.sobel_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y.weight = torch.nn.Parameter(torch.from_numpy(np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])/8.).float().unsqueeze(0).unsqueeze(0))

        gpu = torch.device('cuda')
        self.sobel_x = self.sobel_x.to(gpu)
        self.sobel_y = self.sobel_y.to(gpu)

        self.limit = limit

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
    def __init__(self, crop_fraction=0, vmin=0, vmax=1, limit=10):
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

        self.crop_fraction = crop_fraction
        "Cut-off fraction"

        self.vmin = vmin
        "Lower bound for valid target pixels"

        self.vmax = vmax
        "Upper bound for valid target pixels"

        self.sobel_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_x.weight = torch.nn.Parameter(torch.from_numpy(np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])/8.).float().unsqueeze(0).unsqueeze(0))
        self.sobel_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y.weight = torch.nn.Parameter(torch.from_numpy(np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])/8.).float().unsqueeze(0).unsqueeze(0))

        gpu = torch.device('cuda')
        self.sobel_x = self.sobel_x.to(gpu)
        self.sobel_y = self.sobel_y.to(gpu)

        self.limit = limit

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
        loss = loss + gradloss + 0.03*0.5*uncertainty.sum()/valid_mask.sum().clamp(min=1)

        # if this loss value is not plausible, cap it (which will also not backprop gradients)
        if self.limit is not None and loss > self.limit:
            loss = torch.clamp(loss, max=self.limit)

        if loss.ne(loss).item():
            print("Nan loss!")

        return loss
