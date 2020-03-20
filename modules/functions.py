import torch
import datetime

def prepare_fusion_input(frame, values, weights, config, confidence=None):

    # get frame shape
    b, h, w = frame.shape

    # reshaping data
    values = values.view(b, h, w, config.MODEL.n_points)
    weights = weights.view(b, h, w, config.MODEL.n_points)

    tsdf_frame = torch.unsqueeze(frame, -1)

    # stacking input data
    if config.DATA.confidence:
        assert confidence is not None
        tsdf_confidence = torch.unsqueeze(confidence, -1)
        tsdf_input = torch.cat([tsdf_frame, tsdf_confidence, weights, values], dim=3)
        del tsdf_confidence
    else:
        tsdf_input = torch.cat([tsdf_frame, weights, values], dim=3)

    # permuting input
    tsdf_input = tsdf_input.permute(0, -1, 1, 2)

    return tsdf_input, weights


def prepare_volume_update(data, est, inputs, config, outlier=None, confidence=None):

    tail_points = config.MODEL.n_tail_points
    b, h, w = inputs.shape
    depth = inputs.view(b, h * w, 1)

    if config.LOSS.loss == 'outlier':
        valid = (outlier < 0.5) & (depth != 0.)
        valid = valid.nonzero()[:, 1]

    # filter all uncertain predictions
    elif config.LOSS.loss == 'uncertainty':
        valid = (confidence > 0.8) & (depth != 0.)
        valid = valid.nonzero()[:, 1]

    else:
        valid = (depth != 0.)
        valid = valid.nonzero()[:, 1]

    # moving everything to the cpu
    update_indices = data['indices'].cpu()[:, valid, :tail_points, :, :]
    update_weights = data['weights'].cpu()[:, valid, :tail_points, :]
    update_points = data['points'].cpu()[:, valid, :tail_points, :]
    update_values = est.cpu()[:, valid, :tail_points]

    update_values = torch.clamp(update_values, -0.1, 0.1)

    del valid

    return update_values, update_indices, update_weights, update_points


def routing(batch, routing_model, config):

    """
    Function for the feature extractor, denoising the depth maps and
    correcting outliers

    :param batch: batch with the input data
    :param routing_model: neural model for routing prediction
    :param device: device to train on
    :param config: 3D fusion configuration
    :param routing_config: routing network configuration
    :return: denoised frame
    """

    inputs = batch[config.DATA.input]
    # inputs = inputs.to(device)
    inputs = inputs.unsqueeze_(1)  # add number of channels

    if config.ROUTING.do:

        # if routing_config.MODEL.n_input_channels == 4:
        #     image = batch['image'].float()
        #     image = image.to(device)
        #     inputs = torch.cat((inputs, image), dim=1)

        if config.DATA.dataset == 'microsoft' or \
            config.DATA.dataset == 'eth3d':

            maxd = torch.max(inputs).clone()
            inputs = inputs * 1.3 / maxd
            est = routing_model.forward(inputs)
            frame = est[:, 0, :, :]
            confidence = torch.exp(-1. * est[:, 1, :, :])
            frame = frame * maxd / 1.3

        elif config.DATA.dataset == 'roadsign':
            maxd = torch.max(inputs).clone()
            mind = 1.5
            inputs = torch.where(inputs < mind, torch.zeros_like(inputs), inputs)
            inputs *= (1.2 - 0.4) / (maxd - mind)
            est = routing_model.forward(inputs)
            frame = est[:, 0, :, :]
            confidence = torch.exp(-1. * est[:, 1, :, :])
            frame = frame * (maxd - mind) / (1.2 - 0.4)

        else:
            est = routing_model.forward(inputs)
            frame = est[:, 0, :, :]
            confidence = torch.exp(-1. * est[:, 1, :, :])

        return frame, confidence

    else:
        frame = inputs.squeeze_(1)
        return frame, None


def fusion(input, weights, model, config):

    b, c, h, w = input.shape

    tsdf_pred = model.forward(input)
    tsdf_pred = tsdf_pred.permute(0, 2, 3, 1)

    tsdf_est = tsdf_pred[:, :, :, :config.MODEL.n_points].view(b, h * w, config.MODEL.n_points)

    return tsdf_est


def pipeline(data,
                   entry,
                   routing_network,
                   extractor,
                   fusion_network,
                   integrator,
                   config):

    # routing
    if routing_network is not None:
        frame, confidence = routing(data, routing_network, config)
    else:
        frame, confidence = data[config.DATA.input], None

    # confidence filtering
    if confidence is not None:
        filtered_frame = torch.where(confidence < config.ROUTING.threshold, torch.zeros_like(frame), frame)
    else:
        filtered_frame = frame

    # filtering valid pixels
    mask = data['original_mask']
    frame = torch.where(mask == 0, torch.zeros_like(filtered_frame), filtered_frame)

    # filter boundary pixels
    frame[:, 0:3, :] = 0
    frame[:, -1:-4, :] = 0
    frame[:, :, 0:3] = 0
    frame[:, :, -1:-4] = 0

    # import matplotlib.pyplot as plt
    #
    # plt.imshow(frame.detach().numpy()[0])
    # plt.show()

    # get shape of batch
    b, h, w = frame.shape

    # get current tsdf values
    scene_id = data['scene_id'][0]

    # TODO: check what volume is and change it to interface
    data_est = extractor.forward(frame, data['extrinsics'], data['intrinsics'],
                                 entry['current'], entry['origin'], entry['resolution'], entry['weights'])

    tsdf_input, tsdf_weights = prepare_fusion_input(frame,
                                                    data_est['fusion_values'],
                                                    data_est['fusion_weights'],
                                                    config,
                                                    confidence=confidence)

    tsdf_est = fusion(tsdf_input, tsdf_weights, fusion_network, config)




    update_values, update_indices, \
    update_weights, update_points = prepare_volume_update(data_est,
                                                          tsdf_est,
                                                          frame,
                                                          config)

    values, weights = integrator.forward(update_values,
                                         update_indices,
                                         update_weights,
                                         entry['current'],
                                         entry['weights'])

    return values, weights
