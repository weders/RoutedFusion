import torch

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

    def _fusion(self, input):

        b, c, h, w = input.shape

        tsdf_pred = self._fusion_network.forward(input)
        tsdf_pred = tsdf_pred.permute(0, 2, 3, 1)

        tsdf_est = tsdf_pred[:, :, :,
                             :self.config.MODEL.n_points].view(b,
                                                               h * w,
                                                               self.config.MODEL.n_points)

        return tsdf_est

    def _prepare_fusion_input(self, frame, values, weights, confidence):

        # get frame shape
        b, h, w = frame.shape

        # reshaping data
        values = values.view(b, h, w, self.config.MODEL.n_points)
        weights = weights.view(b, h, w, self.config.MODEL.n_points)

        tsdf_frame = torch.unsqueeze(frame, -1)

        # stacking input data
        if self.config.DATA.confidence:
            assert confidence is not None
            tsdf_confidence = torch.unsqueeze(confidence, -1)
            tsdf_input = torch.cat(
                [tsdf_frame, tsdf_confidence, weights, values], dim=3)
            del tsdf_confidence
        else:
            tsdf_input = torch.cat([tsdf_frame, weights, values], dim=3)

        # permuting input
        tsdf_input = tsdf_input.permute(0, -1, 1, 2)

        return tsdf_input, weights

    def _prepare_volume_update(self, data, est, inputs):

        tail_points = self.config.MODEL.n_tail_points
        b, h, w = inputs.shape
        depth = inputs.view(b, h * w, 1)

        valid = (depth != 0.)
        valid = valid.nonzero()[:, 1]

        # moving everything to the cpu
        update_indices = data['indices'][:, valid, :tail_points, :, :]
        update_weights = data['weights'][:, valid, :tail_points, :]
        update_points = data['points'][:, valid, :tail_points, :]
        update_values = est[:, valid, :tail_points]

        update_values = torch.clamp(update_values, -0.1, 0.1)

        del valid

        return update_values, update_indices, update_weights, update_points

    def forward(self, data, entry):

        # routing network
        if self._routing:
            frame, confidence = self._routing(data)

            # filtering according to confidence map
            frame = torch.where(confidence < self.config.ROUTING.threshold,
                                torch.zeros_like(frame),
                                frame)

        else:
            frame, confidence = data[self.config.DATA.input], None

        # filtering valid pixels
        frame = torch.where(data['original_mask'] == 0,
                            torch.zeros_like(frame),
                            frame)

        # filter boundary pixels
        frame[:, 0:3, :] = 0
        frame[:, -1:-4, :] = 0
        frame[:, :, 0:3] = 0
        frame[:, :, -1:-4] = 0

        canonical_view = self._extractor(frame,
                                         data['extrinsics'],
                                         data['intrinsics'],
                                         entry['current'],
                                         entry['origin'],
                                         entry['resolution'],
                                         entry['weights'])

        tsdf_input, tsdf_weights = self._prepare_fusion_input(frame,
                                                              canonical_view['fusion_values'],
                                                              canonical_view['fusion_weights'],
                                                              confidence)

        tsdf_est = self._fusion(tsdf_input)

        update_values, update_indices, \
        update_weights, update_points = self._prepare_volume_update(canonical_view,
                                                                    tsdf_est,
                                                                    frame)

        values, weights = self._integrator.forward(update_values,
                                                   update_indices,
                                                   update_weights,
                                                   entry['current'],
                                                   entry['weights'])

        return values, weights

    def train(self, data, grid_est, grid_gt):
        raise NotImplementedError
