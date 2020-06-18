import torch

import numpy as np


# class FusionLoss(torch.nn.Module):
#
#     def __init__(self, config):
#
#         super(FusionLoss, self).__init__()
#
#         self.criterion_l1 = torch.nn.L1Loss()
#         self.criterion_l2 = torch.nn.MSELoss()
#         self.criterion_cos = torch.nn.CosineEmbeddingLoss()
#
#         self.w_l1 = config.LOSS.w_l1
#         self.w_l2 = config.LOSS.w_l2
#         self.w_cos = config.LOSS.w_cos
#
#     def forward(self, est, gt):
#
#         x1 = torch.sign(est)
#         x2 = torch.sign(gt)
#
#         label = torch.ones_like(x1)
#
#         l1 = self.criterion_l1(est, gt)
#         l2 = self.criterion_l2(est, gt)
#         lcos = self.criterion_cos(x1, x2, label)
#
#         loss = self.w_l1 * l1 + self.w_l2 * l2 + self.w_cos * lcos
#         return loss


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

        # target = torch.clamp(target, -0.01, 0.01)

        # outlier weighting
        sum = torch.sum(target, dim=-1)

        # n_outliers = torch.sum(sum >= 9*0.01)
        # n_inliers = 76800 - n_outliers

        # if n_outliers == 0:
        #     weights = torch.ones_like(sum).unsqueeze_(-1)
        # else:
        #     pass
            # print('# outliers', n_outliers)

            #
            # weights = torch.where(sum >= 0.9,
            #                       10.*torch.ones_like(sum),
            #                       torch.ones_like(sum)).unsqueeze_(-1)
            #
            # sum = sum.unsqueeze_(-1)
            #
            # est_outlier = masking(est, sum, 0.9, 'geq')
            # target_outlier = masking(target, sum, 0.9, 'geq')
            #
            # x1_outlier = torch.sign(est_outlier)
            # x2_outlier = torch.sign(target_outlier)
            # label_outlier = torch.ones_like(x1_outlier)
            #
            # l1_outlier = self.criterion1.forward(est_outlier, target_outlier)
            # l2_outlier = self.criterion2.forward(est_outlier, target_outlier)
            # l3_outlier = self.criterion3.forward(x1_outlier, x2_outlier, label_outlier)
            #
            # normalization_outlier = label_outlier.sum()
            #
            # l_outlier = l1_outlier.sum() + l2_outlier.sum() + l3_outlier.sum()
            # l_outlier /= normalization_outlier
            #
            # est_inlier = masking(est, sum, 0.9, 'leq')
            # target_inlier = masking(target, sum, 0.9, 'leq')
            #
            # x1_inlier = torch.sign(est_inlier)
            # x2_inlier = torch.sign(target_inlier)
            # label_inlier = torch.ones_like(x1_inlier)
            #
            # l1_inlier = self.criterion1.forward(est_inlier, target_inlier)
            # l2_inlier = self.criterion2.forward(est_inlier, target_inlier)
            # l3_inlier = self.criterion3.forward(x1_inlier, x2_inlier, label_inlier)
            #
            # normalization_inlier = label_inlier.sum()
            #
            # l_inlier = l1_inlier.sum() + l2_inlier.sum() + l3_inlier.sum()
            # l_inlier /= normalization_inlier
            #
            # # print('outlier loss:', l_outlier)
            # print('inlier loss:', l_inlier)
            #
            # print('L1 outlier', l1_outlier.sum() / normalization_outlier)
            # print('L1 inlier', l1_inlier.sum() / normalization_inlier)
            #
            # print('COS outlier', l3_outlier.sum() / normalization_outlier)
            # print('COS inlier', l3_inlier.sum() / normalization_inlier)

        # # geometry weighting
        # geometry_weights = torch.ones_like(target)
        # geometry_weights[:, :, 4:] *= 10.
        #
        # # thin geometry weight
        # thin_geometries = torch.where(target[:, :, -1] > 0, 3*torch.ones_like(target[:, :, -1]), torch.ones_like(target[:, :, -1]))
        #
        # thin_geometries = torch.where(target[:, :, -2] > 0, 3*torch.ones_like(target[:, :, -1]), thin_geometries)
        # thin_geometries = torch.where(target[:, :, -3] > 0, 3*torch.ones_like(target[:, :, -1]), thin_geometries)
        #
        # thin_geometries.unsqueeze_(-1)

        l1 = self.criterion1.forward(est, target)
        l2 = self.criterion2.forward(est, target)
        l3 = self.criterion3.forward(x1, x2, label)

        # l1 = weights * l1
        # l2 = weights * l2
        # l3 = weights * l3

        # l1 = geometry_weights * l1
        # l2 = geometry_weights * l2
        # l3 = geometry_weights * l3

        # l1 = thin_geometries * l1
        # l2 = thin_geometries * l2
        # l3 = thin_geometries * l3

        normalization = torch.ones_like(l1).sum()

        # normalization += weights.sum()
        # normalization += geometry_weights.sum()
        # normalization += thin_geometries.sum()

        l_vis = l1 + l2 + l3
        l_vis /= normalization

        l1 = l1.sum() / normalization
        l2 = l2.sum() / normalization
        l3 = l3.sum() / normalization


        l = self.lambda1*l1 + self.lambda2*l2 + self.lambda3*l3

        #del geometry_weights, thin_geometries, weights

        return l

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
