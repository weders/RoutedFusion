import numpy as np
from copy import copy
from graphics import Voxelgrid

import os
import h5py

from torch.utils.data import Dataset
import torch
from tqdm import tqdm

import datetime

from modules.extractor import interpolation_weights, \
    get_index_mask, extract_values, extract_indices, insert_values, trilinear_interpolation
from modules.extractor import Extractor


class VolumeDB(Dataset):

    def __init__(self, dataset, config):

        super(VolumeDB, self).__init__()

        self.transform = config.transform
        self.initial_value = config.init_value

        self.scenes_gt = {}
        self.scenes_est = {}
        self.fusion_weights = {}
        self.counts = {}

        # load voxelgrids and initialize estimated scenes
        if config.scene_list is not None and dataset.__class__.__name__ != 'ModelNet':
            with open(config.scene_list, 'r') as file:
                scenes = file.readlines()
                for scene in scenes:
                    scene, room = scene.rstrip().split('\t')
                    scene_id = os.path.join(scene, room)
                    self.scenes_gt[scene_id] = dataset.get_grid(scene_id)
                    self.scenes_est[scene_id] = Voxelgrid(self.scenes_gt[scene_id].resolution)
                    self.scenes_est[scene_id].from_array(self.scenes_gt[scene_id].volume,
                                                          self.scenes_gt[scene_id].bbox,)
                    self.fusion_weights[scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape)
                    self.counts[scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape)

        elif dataset.__class__.__name__ == 'Scene3D':
            scene = dataset.scene
            for sp in range(len(dataset.grids)):
                scene_id = '{}.{}'.format(scene, sp)
                self.scenes_gt[scene_id] = dataset.get_grid(sp)
                self.scenes_est[scene_id] = Voxelgrid(self.scenes_gt[scene_id].resolution)
                self.scenes_est[scene_id].from_array(self.scenes_gt[scene_id].volume, self.scenes_gt[scene_id].bbox)
                self.fusion_weights[scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape)
                self.counts[scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape)

        elif dataset.__class__.__name__ == 'Freiburg':

            self.scenes_gt['office'] = dataset.get_grid()
            self.scenes_est['office'] = Voxelgrid(
                self.scenes_gt['office'].resolution)
            self.scenes_est['office'].from_array(
                self.scenes_gt['office'].volume,
                self.scenes_gt['office'].bbox, )
            self.fusion_weights['office'] = np.zeros(
                self.scenes_gt['office'].volume.shape)
            self.counts['office'] = np.zeros(
                self.scenes_gt['office'].volume.shape)

        elif dataset.__class__.__name__ == 'RoadSign':

            self.scenes_gt['roadsign'] = dataset.get_grid()
            self.scenes_est['roadsign'] = Voxelgrid(
                self.scenes_gt['roadsign'].resolution)
            self.scenes_est['roadsign'].from_array(
                self.scenes_gt['roadsign'].volume,
                self.scenes_gt['roadsign'].bbox, )
            self.fusion_weights['roadsign'] = np.zeros(
                self.scenes_gt['roadsign'].volume.shape)
            self.counts['roadsign'] = np.zeros(
                self.scenes_gt['roadsign'].volume.shape)

        elif dataset.__class__.__name__ == 'Microsoft':

            if dataset._config.scene_list is None:
                scene = dataset._scene
                if dataset.split:
                    for sp in range(len(dataset.grids)):
                        scene_id = '{}.{}'.format(scene, sp)
                        self.scenes_gt[scene_id] = dataset.get_grid(sp)
                        self.scenes_est[scene_id] = Voxelgrid(self.scenes_gt[scene_id].resolution)
                        self.scenes_est[scene_id].from_array(self.scenes_gt[scene_id].volume, self.scenes_gt[scene_id].bbox)
                        self.fusion_weights[scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape)
                        self.counts[scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape)
                else:
                    self.scenes_gt[scene] = dataset.get_grid(scene)
                    self.scenes_est[scene] = Voxelgrid(self.scenes_gt[scene].resolution)
                    self.scenes_est[scene].from_array(self.scenes_gt[scene].volume, self.scenes_gt[scene].bbox)
                    self.fusion_weights[scene] = np.zeros(self.scenes_gt[scene].volume.shape)
                    self.counts[scene] = np.zeros(self.scenes_gt[scene].volume.shape)

        elif dataset.__class__.__name__ == 'ETH3D':
            scene = dataset._scene
            for sp in range(len(dataset.grids)):
                scene_id = '{}.{}'.format(scene, sp)
                self.scenes_gt[scene_id] = dataset.get_grid(sp)
                self.scenes_est[scene_id] = Voxelgrid(self.scenes_gt[scene_id].resolution)
                self.scenes_est[scene_id].from_array(self.scenes_gt[scene_id].volume, self.scenes_gt[scene_id].bbox)
                self.fusion_weights[scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape)
                self.counts[scene_id] = np.zeros(self.scenes_gt[scene_id].volume.shape)


        elif dataset.__class__.__name__ == 'Matterport':
            scene = dataset.scene
            self.scenes_gt[scene] = dataset.get_grid()
            self.scenes_est[scene] = Voxelgrid(self.scenes_gt[scene].resolution)
            self.scenes_est[scene].from_array(self.scenes_gt[scene].volume, self.scenes_gt[scene].bbox)
            self.fusion_weights[scene] = np.zeros(self.scenes_gt[scene].volume.shape)
            self.counts[scene] = np.zeros(self.scenes_gt[scene].volume.shape)

        elif dataset.__class__.__name__ == 'Replica':
            scene = dataset.scene
            self.scenes_gt[scene] = dataset.get_grid()
            self.scenes_est[scene] = Voxelgrid(self.scenes_gt[scene].resolution)
            self.scenes_est[scene].from_array(self.scenes_gt[scene].volume, self.scenes_gt[scene].bbox)
            self.fusion_weights[scene] = np.zeros(self.scenes_gt[scene].volume.shape)
            self.counts[scene] = np.zeros(self.scenes_gt[scene].volume.shape)

        elif dataset.__class__.__name__ == 'ModelNet':
            with open(config.scene_list, 'r') as file:
                scenes = file.readlines()
            for scene in scenes:
                key = scene.rstrip()
                key = key.replace('\t', '/')
                self.scenes_gt[key] = dataset.get_grid(key)
                self.scenes_est[key] = Voxelgrid(self.scenes_gt[key].resolution)
                self.scenes_est[key].from_array(self.scenes_gt[key].volume, self.scenes_gt[key].bbox)
                self.fusion_weights[key] = np.zeros(self.scenes_gt[key].volume.shape)
                self.counts[key] = np.zeros(self.scenes_gt[key].volume.shape)


        self.reset()

    def __getitem__(self, item):

        sample = dict()

        sample['gt'] = self.scenes_gt[item].volume
        sample['current'] = self.scenes_est[item].volume
        sample['origin'] = self.scenes_gt[item].origin
        sample['resolution'] = self.scenes_gt[item].resolution
        sample['counts'] = self.counts[item]
        sample['weights'] = self.fusion_weights[item]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.scenes_gt)

    def filter(self, value=2.):

        for key in self.scenes_est.keys():

            weights = self.fusion_weights[key]
            self.scenes_est[key].volume[weights < value] = self.initial_value
            self.fusion_weights[key][weights < value] = 0

    def save(self, path, scene_id=None, epoch=None, groundtruth=False):

        if scene_id is None:
            raise NotImplementedError
        else:
            if epoch is not None:
                filename = '{}.{}.volume.hf5'.format(scene_id.replace('/', '.'),
                                              epoch)
                weightname = '{}.{}.weights.hf5'.format(scene_id.replace('/', '.'),
                                                        epoch)
            else:
                filename = '{}.volume.hf5'.format(scene_id.replace('/', '.'))
                weightname = '{}.weights.hf5'.format(scene_id.replace('/', '.'))

            with h5py.File(os.path.join(path, filename), 'w') as hf:
                hf.create_dataset("TSDF",
                                  shape=self.scenes_est[scene_id].volume.shape,
                                  data=self.scenes_est[scene_id].volume)
            with h5py.File(os.path.join(path, weightname), 'w') as hf:
                hf.create_dataset("weights",
                                  shape=self.fusion_weights[scene_id].shape,
                                  data=self.fusion_weights[scene_id])

            if groundtruth:
                groundtruthname = '{}.gt.hf5'.format(scene_id.replace('/', '.'))
                with h5py.File(os.path.join(path, groundtruthname), 'w') as hf:
                    hf.create_dataset("TSDF",
                                      shape=self.scenes_gt[
                                          scene_id].volume.shape,
                                      data=self.scenes_gt[scene_id].volume)


    def save_txt(self, path, scene_id=None, epoch=None):

        if scene_id is None:
            raise NotImplementedError
        else:
            if epoch is not None:
                filename = '{}.{}.volume.txt'.format(scene_id.replace('/', '.'),
                                              epoch)
                weightname = '{}.{}.weights.txt'.format(scene_id.replace('/', '.'),
                                                        epoch)
            else:
                filename = '{}.volume.txt'.format(scene_id.replace('/', '.'))
                weightname = '{}.weights.txt'.format(scene_id.replace('/', '.'))

            np.savetxt(filename, self.scenes_est[scene_id].volume)
            np.savetxt(weightname, self.fusion_weights[scene_id])

    def reset(self):
        for scene_id in self.scenes_est.keys():
            self.scenes_est[scene_id].volume = self.initial_value*np.ones(self.scenes_est[scene_id].volume.shape)
            self.fusion_weights[scene_id] = np.zeros(self.scenes_est[scene_id].volume.shape)

    def update(self, scene_id,
               coordinates=None,
               values=None,
               valids=None,
               indices=None,
               points=None,
               weights=None,
               mode='ray'):

        if mode == 'ray':

            volume = self.scenes_est[scene_id].volume
            weight_cache = self.fusion_weights[scene_id]
            self.resampling(points, values,
                            weights, indices,
                            volume, weight_cache,
                            scene_id)

        if mode == 'blocks':
            assert voxels is not None
            assert valids is not None
            v1, v2, v3 = np.where(valids == 1)
            coords_valid = coordinates[v1, v2, v3, :]
            self.scenes_est[scene_id].volume[coords_valid[:, 0], coords_valid[:, 1], coords_valid[:, 2]] = values[v1, v2, v3]

        if mode == 'voxel':
            assert indices is not None

            volume = self.scenes_est[scene_id].volume
            self.resampling(points, values, volume)

            # xs, ys, zs = volume.shape
            #
            # b, n_rays, n_voxels = values.shape
            #
            # coords = coordinates.contiguous().view(b * n_rays * n_voxels, 3)
            # voxels = values.view(b * n_rays * n_voxels, 1)
            #
            # valid = ((coords[:, 0] >= 0) &
            #          (coords[:, 0] < xs) &
            #          (coords[:, 1] >= 0) &
            #          (coords[:, 1] < ys) &
            #          (coords[:, 2] >= 0) &
            #          (coords[:, 2] < zs))
            #
            # # get coordinates which are valid
            # x = torch.masked_select(coords[:, 0], valid).long()
            # y = torch.masked_select(coords[:, 1], valid).long()
            # z = torch.masked_select(coords[:, 2], valid).long()
            #
            # values = torch.masked_select(voxels[:, 0], valid)
            #
            # volume[x, y, z] = values.detach().numpy()

    def update_new(self,
                   depth,
                   extrinsics,
                   intrinsics,
                   scene_id):

        scene = self.scenes_est[scene_id]

        origin = scene.origin
        resolution = scene.resolution

        # compute coordinates given new depth map
        coords = compute_coordinates(depth, extrinsics, intrinsics)

        eye_w = extrinsics[:, :3, 3]

        extractor = Extractor()

        ray_pts, ray_dists = extractor.extract_values(coords,
                                                      eye_w.double(),
                                                      torch.Tensor(origin).double(),
                                                      resolution)

        ray_voxels, _, borders = extractor.extract_rays(coords, eye_w.double(),
                                                        torch.Tensor(origin).double(), resolution, gpu=False)

        volume_current = torch.Tensor(self.scenes_est[scene_id].volume).double()
        volume_gt = torch.Tensor(self.scenes_gt[scene_id].volume).double()

        fusion_weights_volume = torch.Tensor(self.fusion_weights[scene_id]).double()


        values_current, values_gt, indices, weights, \
        centers, fusion_weights, values_fused = trilinear_interpolation(ray_pts, volume_current,
                                                                        volume_gt, fusion_weights_volume)

        n1, n2, n3 = values_gt.shape

        indices = indices.view(n1, n2, n3, 8, 3)
        weights = weights.view(n1, n2, n3, 8)

        return coords, ray_pts, indices, weights


    def to_tsdf(self, mode='normal', trunaction=1.2):
        for scene_id in self.scenes_gt.keys():
            self.scenes_gt[scene_id].transform(mode='normal')
            self.scenes_gt[scene_id].volume *= self.scenes_gt[scene_id].resolution
            if mode == 'truncate':
                self.scenes_gt[scene_id].volume[np.abs(self.scenes_gt[scene_id].volume) > trunaction] = self.initial_value

    def _sample_values(self, values, indices):

        n_values = values.shape[-1]

        voxels = torch.zeros(indices.shape)
        ones = torch.ones(voxels.shape)

        for i in range(n_values):
            factor = values[:, :, i].unsqueeze_(2)
            voxels = torch.where(indices == i, factor*ones, voxels)

        return voxels

    def resampling(self, points, values,
                   weights, indices,
                   volume, weight_cache, scene_id, mode='trilinear'):

        if mode == 'nearest':

            b, h, n, dim = points.shape

            # reshaping values and indices for better performance in pytorch
            indices = torch.floor(points).long().view(b*h*n, dim)
            values = values.contiguous().view(b*h*n, 1)

            valid = get_index_mask(indices, volume.shape)
            indices = extract_indices(indices, valid)
            values = torch.masked_select(values[:, 0], valid).detach().numpy()
            weights = np.ones_like(values)

            # inserting values and weights
            insert_values(values, indices, self.scenes_est[scene_id].volume)
            insert_values(weights, indices, self.fusion_weights[scene_id])

        if mode == 'block':

            n1, n2, n3 = indices.shape

            values = values.contiguous().view(n1, 1)
            values = values.repeat(1, 8)

            indices = indices.contiguous().view(n1 * n2, n3).long()
            values = values.contiguous().view(n1 * n2, 1).double()

            valid = get_index_mask(indices, volume.shape)
            indices = extract_indices(indices, valid)
            values = torch.masked_select(values[:, 0], valid).detach().numpy()
            weights = np.ones_like(values)
            insert_values(values, indices, self.scenes_est[scene_id].volume)
            insert_values(weights, indices, self.fusion_weights[scene_id].volume)

        # weights, indices = interpolation_weights(points)

        if mode == 'trilinear':

            xs, ys, zs = volume.shape

            # reshape tensors
            n1, n2, n3 = values.shape

            values = values.contiguous().view(n1*n2*n3, 1)
            values = values.repeat(1, 8)
            indices = indices.contiguous().view(n1*n2*n3, 8, 3).long()
            weights = weights.contiguous().view(n1*n2*n3, 8)

            n1, n2, n3 = indices.shape
            indices = indices.contiguous().view(n1*n2, n3).long()
            weights = weights.contiguous().view(n1*n2, 1).double()
            values = values.contiguous().view(n1*n2, 1).double()

            valid = get_index_mask(indices, volume.shape)

            values = torch.masked_select(values[:, 0], valid)
            indices = extract_indices(indices, mask=valid)
            weights = torch.masked_select(weights[:, 0], valid)

            update = weights*values

            wcache = torch.zeros(volume.shape).double().view(xs*ys*zs)
            vcache = torch.zeros(volume.shape).double().view(xs*ys*zs)

            index = ys*zs*indices[:, 0] + zs*indices[:, 1] + indices[:, 2]

            wcache.index_add_(0, index, weights)
            vcache.index_add_(0, index, update)

            wcache = wcache.view(xs, ys, zs)
            vcache = vcache.view(xs, ys, zs)

            update = extract_values(indices, vcache).detach().numpy()
            weights = extract_values(indices, wcache).detach().numpy()

            values_old = extract_values(indices, volume)
            weights_old = extract_values(indices, weight_cache)

            value_update = (weights_old*values_old + update)/(weights_old + weights)
            # value_update = update

            # print('update', np.unique(update))
            # print('value update', np.unique(value_update))

            weight_update = weights_old + weights

            # value_update = update
            # weight_update = weights_old + weights

            insert_values(value_update, indices, self.scenes_est[scene_id].volume)
            insert_values(weight_update, indices, self.fusion_weights[scene_id])


        # updated_voxel)b, h, n, dim = ray.shape
        # bp, hp, np, dimp = points.shape
        # points = torch.split(points, 1, dim=2)
        #
        # values = values.view(bp, hp, np, 1)
        # values = torch.split(values, 1, dim=2)
        # for i, pt in enumerate(points):
        #
        #     if i < len(points) - 1:
        #         low = pt
        #         high = points[i + 1]
        #         values_low = values[i][0, :, 0, 0].double()
        #         values_high = values[i + 1][0, :, 0, 0].double()
        #     else:
        #         high = pt
        #         low = points[i - 1]
        #         values_low = values[i-1][0, :, 0, 0].double()
        #         values_high = values[i][0, :, 0, 0].double()
        #
        #     to_low = low - ray
        #     to_high = high - ray
        #
        #     dist_to_low = torch.norm(to_low, dim=3)
        #     dist_to_high = torch.norm(to_high, dim=3)
        #
        #     sign = torch.sum(torch.sign(to_low) - torch.sign(to_high), dim=3)
        #     same_side = torch.where(sign != 0.,
        #                             torch.zeros_like(sign).byte(),
        #                             torch.ones_like(sign).byte())
        #
        #
        #
        #     coords = ray.view(b * h * n, dim)
        #
        #     updated_voxel = 0
        #
        #     if i == 0:
        #
        #         cond1 = same_side
        #         cond2 = (dist_to_low < dist_to_high)
        #         cond3 = (dist_to_low < 0.25*torch.ones_like(dist_to_low))
        #
        #         cond = cond1 & cond2 & cond3
        #
        #         same_side = torch.where(cond,
        #                                 torch.ones_like(same_side).byte(),
        #                                 torch.zeros_like(same_side).byte())
        #
        #         mask = same_side.view(b*h*n, )
        #
        #         update_values = values_low.repeat(1, n)
        #         update_values = update_values.view(b*h*n, 1)
        #
        #         coords_chunk = torch.chunk(coords, 1, dim=0)
        #         mask_chunk = torch.chunk(mask, 1, dim=0)
        #         update_chunk = torch.chunk(update_values, 1, dim=0)
        #
        #         for c, m, v in zip(coords_chunk, mask_chunk, update_chunk):
        #
        #             x = torch.floor(torch.masked_select(c[:, 0], m)).long()
        #             y = torch.floor(torch.masked_select(c[:, 1], m)).long()
        #             z = torch.floor(torch.masked_select(c[:, 2], m)).long()
        #
        #             update = torch.masked_select(v[:, 0], m)
        #
        #             valid = (0 <= x) & (x < xs)
        #             valid = valid & (0 <= y) & (y < ys)
        #             valid = valid & (0 <= z) & (z < zs)
        #
        #             x = torch.masked_select(x, valid)
        #             y = torch.masked_select(y, valid)
        #             z = torch.masked_select(z, valid)
        #             update = torch.masked_select(update, valid)
        #
        #             values_old = torch.Tensor(volume[x, y, z]).double()
        #
        #             volume[x, y, z] = update
        #
        #             # volume[x, y, z] = torch.where(values_old == 10.e7,
        #             #                               update,
        #             #                               0.5*(update + values_old))
        #
        #             updated_voxel += torch.sum(m)
        #
        #
        #     elif i == len(points) - 1:
        #
        #         cond1 = same_side
        #         cond2 = (dist_to_low > dist_to_high)
        #         cond3 = (dist_to_high < 0.25 * torch.ones_like(dist_to_low))
        #
        #         cond = cond1 & cond2 & cond3
        #
        #         same_side = torch.where(cond,
        #                                 torch.ones_like(same_side).byte(),
        #                                 torch.zeros_like(same_side).byte())
        #
        #         mask = same_side.view(b * h * n, )
        #
        #         update_values = values_high.repeat(1, n)
        #         update_values = update_values.view(b * h * n, 1)
        #
        #         coords_chunk = torch.chunk(coords, 1, dim=0)
        #         mask_chunk = torch.chunk(mask, 1, dim=0)
        #         update_chunk = torch.chunk(update_values, 1, dim=0)
        #
        #         for c, m, v in zip(coords_chunk, mask_chunk, update_chunk):
        #             x = torch.floor(torch.masked_select(c[:, 0], m)).long()
        #             y = torch.floor(torch.masked_select(c[:, 1], m)).long()
        #             z = torch.floor(torch.masked_select(c[:, 2], m)).long()
        #
        #             update = torch.masked_select(v[:, 0], m)
        #
        #             valid = (0 <= x) & (x < xs)
        #             valid = valid & (0 <= y) & (y < ys)
        #             valid = valid & (0 <= z) & (z < zs)
        #
        #             x = torch.masked_select(x, valid)
        #             y = torch.masked_select(y, valid)
        #             z = torch.masked_select(z, valid)
        #             update = torch.masked_select(update, valid)
        #
        #             values_old = torch.Tensor(volume[x, y, z]).double()
        #
        #             volume[x, y, z] = update
        #
        #             # volume[x, y, z] = torch.where(values_old == 10.e7,
        #             #                               update,
        #             #                               0.5 * (update + values_old))
        #
        #             updated_voxel += torch.sum(m)
        #
        #
        #         continue
        #
        #
        #
        #     mask = torch.ones_like(same_side) - same_side
        #     print('same_side', torch.sum(same_side))
        #     update = values_low.unsqueeze_(-1)*dist_to_low.squeeze_(0) + \
        #              values_high.unsqueeze_(-1)*dist_to_high.squeeze_(0)
        #     update = update/(dist_to_low + dist_to_high)
        #
        #     mask = mask.view(b*h*n, 1)
        #     update = update.view(b*h*n, 1)
        #
        #     x = torch.floor(torch.masked_select(coords[:, 0], mask[:, 0])).long()
        #     y = torch.floor(torch.masked_select(coords[:, 1], mask[:, 0])).long()
        #     z = torch.floor(torch.masked_select(coords[:, 2], mask[:, 0])).long()
        #
        #     update = torch.masked_select(update[:, 0], mask[:, 0])
        #
        #     valid = (0 <= x) & (x < xs)
        #     valid = valid & (0 <= y) & (y < ys)
        #     valid = valid & (0 <= z) & (z < zs)
        #
        #     x = torch.masked_select(x, valid)
        #     y = torch.masked_select(y, valid)
        #     z = torch.masked_select(z, valid)
        #
        #     update = torch.masked_select(update, valid)
        #
        #     values_old = torch.Tensor(volume[x, y, z]).double()
        #
        #     volume[x, y, z] = update
        #
        #     # volume[x, y, z] = torch.where(values_old == 10.e7,
        #     #                               update,
        #     #                               0.5 * (update + values_old))
        #
        #     updated_voxel += torch.sum(mask)
        #
        # print('# updated voxels:',

    def _update_volume(self, scene_id, voxels, coords, indices):

        volume = self.scenes_est[scene_id].volume
        xs, ys, zs = volume.shape

        b, n_rays, n_voxels = voxels.shape

        coords = coords.contiguous().view(b*n_rays*n_voxels, 3)
        indices = indices.contiguous().view(b*n_rays*n_voxels, 1)
        voxels = voxels.view(b*n_rays*n_voxels, 1)

        valid = ((coords[:, 0] >= 0) &
                 (coords[:, 0] < xs) &
                 (coords[:, 1] >= 0) &
                 (coords[:, 1] < ys) &
                 (coords[:, 2] >= 0) &
                 (coords[:, 2] < zs) &
                 (indices[:, 0] != -1.))

        # get coordinates which are valid
        x = torch.masked_select(coords[:, 0], valid).long()
        y = torch.masked_select(coords[:, 1], valid).long()
        z = torch.masked_select(coords[:, 2], valid).long()

        values = torch.masked_select(voxels[:, 0], valid)

        volume[x, y, z] = values.detach().numpy()


    def compare(self, scene_id):

        est = self.scenes_est[scene_id].volume
        self.scenes_gt[scene_id].compare(reference=est)

    def carving(self, scene_id, frame, extrinsics, intrinsics):

        b, h, w = frame.shape

        points = compute_coordinates(frame, extrinsics, intrinsics)
        camera = extrinsics[:3, 3]

        for p in points:

            carve(self.scenes_est[scene_id].volume, camera, p)


# def carve(volume, start, end):
#


def compute_coordinates(depth, extrinsics, intrinsics):

    b, h, w = depth.shape

    xx, yy = np.meshgrid(np.arange(0, h), np.arange(0, w))

    xx = np.reshape(xx, (b, h*w, 1))
    yy = np.reshape(yy, (b, h*w, 1))
    zz = np.reshape(depth.cpu().detach().numpy(), (b, h*w, 1))

    points_p = np.concatenate((yy, xx, zz), axis=2)

    points_p[:, :, 0] *= zz[:, :, 0]
    points_p[:, :, 1] *= zz[:, :, 0]

    points_p = np.transpose(points_p, axes=(0, 2, 1))
    points_c = np.matmul(np.linalg.inv(intrinsics), points_p)
    points_c = np.concatenate((points_c, np.ones((b, 1, h*w))), axis=1)
    points_w = np.matmul(extrinsics[:3], points_c)
    points_w = np.transpose(points_w, axes=(0, 2, 1))[:, :, :3]

    return points_w


if __name__ == '__main__':

    from datasets.dataset import SUNCG
    from weight_learning.config_end_to_end import config

    train_data = SUNCG(scene_list='weight_learning/list/suncg/debug_v1.2.txt',
                       root_dir='~/Polybox/Master/03-semester/01-master-thesis/05-data/01-suncg/suncg_v1',
                       keys=config.DATA.modalities,
                       n_samples=config.TRAINING.n_samples)

    train_tsdfdb = VolumeDB(scene_list='weight_learning/list/suncg/debug_v1.2.txt',
                            dataset=train_data)

    volume = train_tsdfdb.scenes_gt['000d0395709d2a16e195c6f0189155c4/room_fr_0rm_0'].volume

    train_tsdfdb.to_tsdf()

    tsdf = train_tsdfdb.scenes_gt['000d0395709d2a16e195c6f0189155c4/room_fr_0rm_0'].volume
