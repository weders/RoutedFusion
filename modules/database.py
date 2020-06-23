import os
import h5py

import numpy as np

from torch.utils.data import Dataset
from graphics import Voxelgrid

from utils.metrics import evaluation


class Database(Dataset):

    def __init__(self, dataset, config):

        super(Database, self).__init__()

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

    def save_to_workspace(self, workspace):

        for key in self.scenes_est.keys():

            tsdf_volume = self.scenes_est[key].volume
            weight_volume = self.fusion_weights[key]

            tsdf_file = key.replace('/', '.') + '.tsdf.hf5'
            weight_file = key.replace('/', '.') + '.weights.hf5'

            workspace.save_tsdf_data(tsdf_file, tsdf_volume)
            workspace.save_weigths_data(weight_file, weight_volume)

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

    def evaluate(self):

        eval_results = {}

        for scene_id in self.scenes_est.keys():

            print('Evaluating ', scene_id, '...')

            weights = self.fusion_weights[scene_id]
            est = self.scenes_est[scene_id].volume
            gt = self.scenes_gt[scene_id].volume

            mask = np.copy(weights)
            mask[mask > 0] = 1.

            eval_results_scene = evaluation(est, gt, mask)

            for key in eval_results_scene.keys():

                print(key, eval_results_scene[key])

                if not eval_results.get(key):
                    eval_results[key] = eval_results_scene[key]
                else:
                    eval_results[key] += eval_results_scene[key]



        # normalizing metrics
        for key in eval_results.keys():
            eval_results[key] /= len(self.scenes_est.keys())

        return eval_results

    def reset(self):
        for scene_id in self.scenes_est.keys():
            self.scenes_est[scene_id].volume = self.initial_value * np.ones(self.scenes_est[scene_id].volume.shape)
            self.fusion_weights[scene_id] = np.zeros(self.scenes_est[scene_id].volume.shape)

    def to_tsdf(self, mode='normal', truncation=1.2):
        for scene_id in self.scenes_gt.keys():
            self.scenes_gt[scene_id].transform(mode='normal')
            self.scenes_gt[scene_id].volume *= self.scenes_gt[scene_id].resolution
            if mode == 'truncate':
                self.scenes_gt[scene_id].volume[np.abs(self.scenes_gt[scene_id].volume) > truncation] = self.initial_value
