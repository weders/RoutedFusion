import torch
import datetime

from modules.routing import ConfidenceRouting
from modules.extractor import Extractor
from modules.model import FusionNet
from modules.integrator import Integrator


class Pipeline(torch.nn.Module):

    def __init__(self, config):

        super(Pipeline, self).__init__()

        self.config = config

        if config.ROUTING.do:
            self._routing_network = ConfidenceRouting(1, 64, 1, 1, False)
        else:
            self._routing_network = None

        self._extractor = Extractor(config.MODEL)
        self._fusion_network = FusionNet(config.MODEL)
        self._integrator = Integrator(config.MODEL)

    def _routing(self, data):

        inputs = data[self.config.DATA.input]
        # inputs = inputs.to(device)
        inputs = inputs.unsqueeze_(1)  # add number of channels
        est = self._routing_network.forward(inputs)
        frame = est[:, 0, :, :]
        confidence = torch.exp(-1. * est[:, 1, :, :])

        return frame, confidence

    def _fusion(self, input, weights, values):

        b, c, h, w = input.shape

        tsdf_pred = self._fusion_network.forward(input)
        tsdf_pred = tsdf_pred.permute(0, 2, 3, 1)

        output = dict()

        tsdf_est = tsdf_pred[:, :, :, :self.config.MODEL.n_points].view(b, h * w,
                                                                   self.config.MODEL.n_points)

        # computing weighted updates for loss calculation
        tsdf_old = values['fusion_values']
        tsdf_new = torch.clamp(tsdf_est,
                               -self.config.DATA.init_value,
                               self.config.DATA.init_value)
        weights = weights.view(b, h * w, self.config.MODEL.n_points)
        weights = torch.where(weights < 0, torch.zeros_like(weights), weights)

        tsdf_fused = (weights * tsdf_old + tsdf_new) / (
                    weights + torch.ones_like(weights))

        output['tsdf_est'] = tsdf_est
        output['tsdf_fused'] = tsdf_fused

        return output

    def _prepare_fusion_input(self, frame, values, confidence=None):

        # get frame shape
        b, h, w = frame.shape

        # extracting data
        tsdf_input = values['fusion_values']
        tsdf_weights = values['fusion_weights']

        # reshaping data
        tsdf_input = tsdf_input.view(b, h, w, self.config.MODEL.n_points)
        tsdf_weights = tsdf_weights.view(b, h, w, self.config.MODEL.n_points)

        tsdf_frame = torch.unsqueeze(frame, -1)

        # stacking input data
        if self.config.MODEL.confidence:
            assert confidence is not None
            tsdf_confidence = torch.unsqueeze(confidence, -1)
            tsdf_input = torch.cat([tsdf_frame, tsdf_confidence, tsdf_weights, tsdf_input], dim=3)
            del tsdf_confidence
        else:
            tsdf_input = torch.cat([tsdf_frame, tsdf_weights, tsdf_input], dim=3)

        # permuting input
        tsdf_input = tsdf_input.permute(0, -1, 1, 2)

        del tsdf_frame

        return tsdf_input, tsdf_weights#, tsdf_target

    def _prepare_volume_update(self, values, est, inputs):

        tail_points = self.config.MODEL.n_tail_points

        b, h, w = inputs.shape
        depth = inputs.view(b, h * w, 1)

        valid = (depth != 0.)
        valid = valid.nonzero()[:, 1]

        update_indices = values['indices'][:, valid, :tail_points, :, :]
        update_weights = values['weights'][:, valid, :tail_points, :]
        update_points = values['points'][:, valid, :tail_points, :]
        update_values = est[:, valid, :tail_points]

        update_values = torch.clamp(update_values,
                                    -self.config.DATA.init_value,
                                    self.config.DATA.init_value)

        del valid

        return update_values, update_indices, update_weights, update_points

    def fuse(self,
             batch,
             database,
             device):

        # routing
        if self.config.ROUTING.do:
            frame, confidence = self._routing(batch)
        else:
            frame = batch[self.config.DATA.input].squeeze_(1)
            frame = frame.to(device)
            confidence = None

        mask = batch['original_mask'].to(device)
        filtered_frame = torch.where(mask == 0, torch.zeros_like(frame),
                                     frame)

        # get current tsdf values
        scene_id = batch['scene_id'][0]
        volume = database[scene_id]

        values = self._extractor.forward(frame,
                                         batch['extrinsics'],
                                         batch['intrinsics'],
                                         volume['current'],
                                         volume['origin'],
                                         volume['resolution'],
                                         volume['weights'])


        tsdf_input, tsdf_weights = self._prepare_fusion_input(frame, values,
                                                              confidence)



        fusion_output = self._fusion(tsdf_input, tsdf_weights, values)



        # reshaping target

        # masking invalid losses
        tsdf_est = fusion_output['tsdf_est']

        values['points'] = values['points'][:, :, :self.config.MODEL.n_points].contiguous()

        update_values, update_indices, \
        update_weights, update_points = self._prepare_volume_update(values,
                                                                    tsdf_est,
                                                                    filtered_frame)

        values, weights = self._integrator.forward(update_values.to(device),
                                                   update_indices.to(device),
                                                   update_weights.to(device),
                                                   database[
                                                       batch['scene_id'][0]][
                                                       'current'].to(device),
                                                   database[
                                                       batch['scene_id'][0]][
                                                       'weights'].to(device))


        database.scenes_est[
            batch['scene_id'][0]].volume = values.cpu().detach().numpy()
        database.fusion_weights[
            batch['scene_id'][0]] = weights.cpu().detach().numpy()

        return

    def fuse_training(self, batch, database, device):

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
        if self.config.ROUTING.do:
            frame, confidence = self._routing(batch)
        else:
            frame = batch[self.config.DATA.input].squeeze_(1)
            frame = frame.to(device)
            confidence = None

        mask = batch['original_mask'].to(device)
        filtered_frame = torch.where(mask == 0, torch.zeros_like(frame), frame)

        b, h, w = frame.shape

        # get current tsdf values
        scene_id = batch['scene_id'][0]
        volume = database[scene_id]


        values = self._extractor.forward(frame,
                                         batch['extrinsics'],
                                         batch['intrinsics'],
                                         volume['current'],
                                         volume['origin'],
                                         volume['resolution'],
                                         volume['weights'])

        values_gt = self._extractor.forward(frame,
                                         batch['extrinsics'],
                                         batch['intrinsics'],
                                         volume['gt'],
                                         volume['origin'],
                                         volume['resolution'],
                                         volume['weights'])


        tsdf_input, tsdf_weights = self._prepare_fusion_input(frame, values, confidence)

        tsdf_target = values_gt['fusion_values']
        tsdf_target = tsdf_target.view(b, h, w, self.config.MODEL.n_points)

        fusion_output = self._fusion(tsdf_input, tsdf_weights, values)

        # reshaping target
        tsdf_target = tsdf_target.view(b, h * w, self.config.MODEL.n_points)

        # masking invalid losses
        tsdf_est = fusion_output['tsdf_est']
        tsdf_fused = fusion_output['tsdf_fused']

        tsdf_fused = masking(tsdf_fused, filtered_frame.view(b, h * w, 1))
        tsdf_target = masking(tsdf_target, filtered_frame.view(b, h * w, 1))

        #values['gt'] = values['gt'][:, :, :self.config.MODEL.n_points].contiguous()
        values['points'] = values['points'][:, :, :self.config.MODEL.n_points].contiguous()

        update_values, update_indices, \
        update_weights, update_points = self._prepare_volume_update(values,
                                                                    tsdf_est,
                                                                    filtered_frame)

        values, weights = self._integrator.forward(update_values.to(device),
                                                   update_indices.to(device),
                                                   update_weights.to(device),
                                                   database[batch['scene_id'][0]][
                                                       'current'].to(device),
                                                   database[batch['scene_id'][0]][
                                                         'weights'].to(device))

        database.scenes_est[
            batch['scene_id'][0]].volume = values.cpu().detach().numpy()
        database.fusion_weights[
            batch['scene_id'][0]] = weights.cpu().detach().numpy()

        output['tsdf_est'] = fusion_output['tsdf_est']
        output['tsdf_fused'] = tsdf_fused
        output['tsdf_target'] = tsdf_target

        del update_values, update_points, update_indices, update_weights, values
        return output


def masking(x, values, threshold=0., option='ueq'):

    if option == 'leq':

        if x.dim() == 2:
            valid = (values <= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values <= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == 'geq':

        if x.dim() == 2:
            valid = (values >= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values >= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == 'eq':

        if x.dim() == 2:
            valid = (values == threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values == threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == 'ueq':

        if x.dim() == 2:
            valid = (values != threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values != threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]


    return xmasked
