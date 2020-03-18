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

    print(est.shape)

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

    print(update_values.shape)

    update_values = torch.clamp(update_values, -0.1, 0.1)

    del valid

    return update_values, update_indices, update_weights, update_points


def routing(batch, routing_model, device, config, routing_config):

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
    inputs = inputs.to(device)
    inputs = inputs.unsqueeze_(1)  # add number of channels

    if config.ROUTING.do:

        if routing_config.MODEL.n_input_channels == 4:
            image = batch['image'].float()
            image = image.to(device)
            inputs = torch.cat((inputs, image), dim=1)

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


def pipeline_clean(data,
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






def pipeline(batch,
             extractor, routing_model, tsdf_model, integrator, database,
             device, config, routing_config, sigma, mean, mode='test'):

    """
    Learned real-time depth map fusion pipeline

    :param batch:
    :param extractor:
    :param routing_model:
    :param tsdf_model:
    :param database:
    :param device:
    :param config:
    :param routing_config:
    :param mode:
    :return:
    """
    output = dict()

    # routing
    if routing_model is not None:
        frame, confidence = routing(batch, routing_model, device, config, routing_config)
    else:
        frame = batch[config.DATA.input]
        confidence = None

    if confidence is not None:
        confidence = confidence.clone()
        # filter according to confidence

        if config.DATA.dataset == 'scene3d':
            filtered_frame = torch.where(confidence > 0.8, frame, torch.zeros_like(frame))
            mask = batch['mask'].to(device)
            filtered_frame = torch.where(mask == 0, torch.zeros_like(filtered_frame), filtered_frame)

        elif config.DATA.dataset == 'microsoft' or config.DATA.dataset == 'eth3d':
            filtered_frame = torch.where(confidence > 0.8, frame, torch.zeros_like(frame))
            mask = batch['mask'].to(device)
            filtered_frame = torch.where(mask == 0, torch.zeros_like(filtered_frame), filtered_frame)

        elif config.DATA.dataset == 'roadsign':
            filtered_frame = frame
            # filtered_frame = torch.where(confidence > 0.7, frame, torch.zeros_like(frame))
            mask = batch['mask'].to(device)
            filtered_frame = torch.where(mask == 0, torch.zeros_like(filtered_frame), filtered_frame)


        elif config.DATA.dataset == 'modelnet' or config.DATA.dataset == 'shapenet':
            mask = batch['original_mask'].to(device)
            original_frame = batch[config.DATA.input].to(device)
            original_frame = torch.where(mask == 0, torch.zeros_like(frame), original_frame)

            output['frame'] = frame
            output['confidence'] = confidence

            filtered_frame = torch.where(confidence < config.ROUTING.threshold, torch.zeros_like(frame), frame)
            filtered_frame = torch.where(mask == 0, torch.zeros_like(filtered_frame), filtered_frame)

            # filter boundary pixels
            filtered_frame[:, 0:3, :] = 0
            filtered_frame[:, -1:-4, :] = 0
            filtered_frame[:, :, 0:3] = 0
            filtered_frame[:, :, -1:-4] = 0

            # import matplotlib.pyplot as plt
            # plt.imshow(filtered_frame.cpu().detach().numpy()[0])
            # plt.show()
            #
            # target_frame = batch[config.DATA.target].detach().numpy()[0]
            # plt.imshow(np.abs(filtered_frame.cpu().detach().numpy()[0] - target_frame))
            # plt.show()

            # filtered_frame = filtered_frame.squeeze_(1).cpu().detach().numpy()
            # filtered_frame = median_filter(filtered_frame, 3)
            # filtered_frame = torch.Tensor(filtered_frame).to(device)
            #
            # filtered_frame = torch.where(confidence < 0.95, torch.zeros_like(frame), filtered_frame)
            # filtered_frame = torch.where(mask == 0, torch.zeros_like(filtered_frame), filtered_frame)
        else:
            filtered_frame = torch.where(confidence > config.ROUTING.threshold, frame, torch.zeros_like(frame))



    else:

        if config.DATA.dataset == 'modelnet' or config.DATA.dataset == 'shapenet':
            mask = batch['original_mask']
            filtered_frame = torch.where(mask == 0, torch.zeros_like(frame), frame)

        else:
            filtered_frame = frame
            # import matplotlib.pyplot as plt
            # plt.imshow(filtered_frame.cpu().detach().numpy()[0])
            # plt.show()

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(filtered_frame[0].cpu().detach().numpy())
    # ax[1].imshow(np.abs(filtered_frame[0, 0].cpu().detach().numpy() - batch['depth'].detach().numpy()[0]))
    # ax[2].imshow(confidence.cpu().detach().numpy()[0])
    # plt.show()

    # get shape of batch
    b, h, w = frame.shape

    # get current tsdf values
    scene_id = batch['scene_id'][0]
    volume = database[scene_id]


    if config.MODEL.dynamic:

        frame_gt = batch[config.DATA.target]
        frame_gt = frame_gt.to(device)

        values_gt = extractor.forward(frame_gt,
                                      batch['extrinsics'],
                                      batch['intrinsics'],
                                      volume)

        values_est = extractor.forward(frame,
                                       batch['extrinsics'],
                                       batch['intrinsics'],
                                       volume)

        values = dict()

        values['current'] = values_est['current']
        values['gt'] = values_gt['gt']
        values['fusion_weights'] = values_est['fusion_weights']
        values['points'] = values_est['points']
        values['indices'] = values_est['indices']
        values['weights'] = values_est['weights']

    else:
        values = extractor.forward(frame,
                                   batch['extrinsics'],
                                   batch['intrinsics'],
                                   volume)

    groundtruth = values['gt']
    groundtruth = torch.clamp(groundtruth, -0.01, 0.01)

    outlier = torch.sum(groundtruth, dim=-1)
    outlier = torch.abs(outlier)
    outlier = outlier.view(b, h, w)
    outlier = torch.where(outlier >= 7 * 0.01, torch.ones_like(outlier), torch.zeros_like(outlier))
    # print(torch.sum(outlier).item())
    output['outlier'] = outlier
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(outlier[0].cpu().detach().numpy())
    # ax[1].imshow(filtered_frame.cpu().detach().numpy()[0])
    # plt.show()

    tsdf_input, tsdf_weights, tsdf_target, mean, sigma = prepare_fusion_input(frame, values, config, confidence=confidence, sigma=sigma, mean=mean)

    fusion_output = fusion(tsdf_input, tsdf_weights, tsdf_model, values, config)

    # reshaping target
    tsdf_target = tsdf_target.view(b, h * w, config.MODEL.n_points)

    # # computing outlier ground-truth
    # target_ray_sum = torch.sum(tsdf_target, dim=2)
    # outliers = torch.where(target_ray_sum > 1.5,
    #                        torch.ones_like(target_ray_sum),
    #                        torch.zeros_like(target_ray_sum))
    # outliers = outliers.view(b, h * w, 1)

    if mode == 'train' or mode == 'val':

        if config.DATA.dataset == 'modelnet' or config.DATA.dataset == 'shapenet':
            tsdf_target_unmasked = masking(tsdf_target, batch['original_mask'].to(device).view(b, h*w, 1))
            tsdf_fused_unmasked = masking(fusion_output['tsdf_fused'].clone(), batch['original_mask'].to(device).view(b, h*w, 1))

            output['tsdf_target_unmasked'] = tsdf_target_unmasked
            output['tsdf_fused_unmasked'] = tsdf_fused_unmasked
        else:
            output['tsdf_target_unmasked'] = tsdf_target
            output['tsdf_fused_unmasked'] = fusion_output['tsdf_fused'].clone()

    # masking invalid losses
    tsdf_est = fusion_output['tsdf_est']
    tsdf_fused = fusion_output['tsdf_fused']

    loss = torch.abs(tsdf_fused - groundtruth)
    loss = torch.sum(loss, dim=-1)
    loss = loss.view(b, h, w)
    loss = torch.where(filtered_frame == 0, torch.zeros_like(loss), loss)

    output['loss'] = loss

    # import matplotlib.pyplot as plt
    # plt.imshow(loss[0].cpu().detach().numpy())
    # plt.show()

    tsdf_fused = masking(tsdf_fused, filtered_frame.view(b, h * w, 1))
    tsdf_target = masking(tsdf_target, filtered_frame.view(b, h * w, 1))

    # tsdf_outliers = masking(outliers, filtered_frame.view(b, h * w, 1))

    # """Computing Losses"""
    #
    if config.LOSS.loss == 'uncertainty':
        tsdf_conf = torch.exp(-1. * fusion_output['tsdf_unc'])
        # tsdf_unc = masking(tsdf_unc, filtered_frame.view(b, h * w, 1))

    elif config.LOSS.loss == 'outlier':
        tsdf_unc = fusion_output['tsdf_unc']
        # tsdf_unc = masking(fusion_output['tsdf_unc'], filtered_frame.view(b, h * w, 1))

    values['gt'] = values['gt'][:, :, :config.MODEL.n_points].contiguous()
    values['points'] = values['points'][:, :, :config.MODEL.n_points].contiguous()

    # del tsdf_est, est, tsdf_input, inputs, values
    # rescaling
    # tsdf_update_est = sigma * tsdf_est + mean

    if config.MODEL.dynamic:
        depth_new = fusion_output['tsdf_depth']

        values_new = extractor.forward(depth_new,
                                       intrinsics=batch['intrinsics'],
                                       extrinsics=batch['extrinsics'],
                                       volume=volume)

        values['points'] = values_new['points']
        values['indices'] = values_new['indices']
        values['weights'] = values_new['weights']

        output['tsdf_depth'] = fusion_output['tsdf_depth']

        update_values, update_indices, update_weights, update_points = prepare_volume_update(values,
                                                                                             tsdf_est,
                                                                                             filtered_frame,
                                                                                             config)



    else:


        if config.LOSS.loss == 'outlier':
            print('outliers befure update preparation', torch.sum(torch.where(tsdf_unc > 0.5, torch.ones_like(tsdf_unc), torch.zeros_like(tsdf_unc))))
            update_values, update_indices, update_weights, update_points = prepare_volume_update(values,
                                                                                                 tsdf_est,
                                                                                                 filtered_frame,
                                                                                                 config,
                                                                                                 outlier=tsdf_unc)

            print('update shape', update_values.shape)

        elif config.LOSS.loss == 'uncertainty':
            update_values, update_indices, update_weights, update_points = prepare_volume_update(values,
                                                                                                 tsdf_est,
                                                                                                 filtered_frame,
                                                                                                 config,
                                                                                                 confidence=tsdf_conf)

        else:
            update_values, update_indices, update_weights, update_points = prepare_volume_update(values,
                                                                                                 tsdf_est,
                                                                                                 filtered_frame,
                                                                                                 config)
    print(update_indices.shape)

    test_values = update_values[:, 0, :].unsqueeze_(1)
    test_indices = update_indices[:, 0, :, :, :].unsqueeze_(1)
    test_weights = update_weights[:, 0, :, :].unsqueeze_(1)

    values, weights = integrator.forward(update_values,
                                         update_indices,
                                         update_weights,
                                         database[batch['scene_id'][0]]['current'],
                                         database[batch['scene_id'][0]]['weights'])

    database.scenes_est[batch['scene_id'][0]].volume = values.cpu().detach().numpy()
    database.fusion_weights[batch['scene_id'][0]] = weights.cpu().detach().numpy()

    if mode == 'test':
        return

    elif mode == 'val' or mode == 'train':

        output['tsdf_est'] = fusion_output['tsdf_est']
        output['tsdf_fused'] = tsdf_fused
        output['tsdf_target'] = tsdf_target

        if config.LOSS.loss == 'uncertainty':
            tsdf_conf_unmasked = tsdf_conf.clone()
            tsdf_unc_unmasked = fusion_output['tsdf_unc'].clone()

            tsdf_unc = masking(tsdf_unc_unmasked, filtered_frame.view(b, w * h, 1))
            tsdf_conf = masking(tsdf_conf_unmasked, filtered_frame.view(b, w * h, 1))

            output['tsdf_conf_unmasked'] = tsdf_conf_unmasked

            output['tsdf_unc'] = tsdf_unc
            output['tsdf_conf'] = tsdf_conf

        if config.LOSS.loss == 'outlier':
            if config.DATA.dataset == 'modelnet' or config.DATA.dataset == 'shapenet':
                tsdf_unc = masking(tsdf_unc, batch['original_mask'].to(device).view(b, h * w, 1))
            output['tsdf_outliers'] = tsdf_unc

        if config.ROUTING.do:
            output['depth'] = frame
            output['filtered_frame'] = filtered_frame
            output['uncertainty'] = -torch.log(confidence)
            output['confidence'] = confidence

        del update_values, update_points, update_indices, update_weights, values
        return output
