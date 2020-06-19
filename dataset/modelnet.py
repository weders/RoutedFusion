import os
import glob
import cc3d

from skimage import io, transform
from torch.utils.data import Dataset
from copy import copy

from graphics import Voxelgrid

from scipy.ndimage.morphology import binary_dilation

from utils.data import *

from dataset.binvox_utils import read_as_3d_array


class ModelNet(Dataset):

    def __init__(self, config):

        self.root_dir = os.path.expanduser(config.root_dir)

        self.resolution = (config.resy, config.resx)
        self.xscale = self.resolution[0] / 480.
        self.yscale = self.resolution[1] / 640.

        self.transform = config.transform

        self.scene_list = config.scene_list

        self.noise_scale = config.noise_scale
        self.outlier_scale = config.outlier_scale
        self.outlier_fraction = config.outlier_fraction

        self.grid_resolution = config.grid_resolution

        self.split = 'test'

        self._load_frames()

    def _load_frames(self):

        self._scenes = []
        self.frames = []

        with open(self.scene_list, 'r') as file:

            for line in file:
                try:
                    scene, obj = line.rstrip().split('\t')
                except:
                    scene, obj = line.rstrip().split(' ')

                self._scenes.append(os.path.join(scene, obj))

                path = os.path.join(self.root_dir, scene, self.split, obj,
                                    'data', '*.depth.png')
                files = glob.glob(path)
                for i, f in enumerate(files):
                        self.frames.append(f.replace('.depth.png', ''))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        frame = self.frames[item]

        pathsplit = frame.split('/')
        sc = pathsplit[-5]
        obj = pathsplit[-3]
        scene_id = '{}/{}'.format(sc, obj)
        sample = {}

        frame_id = frame.split('/')[-1]
        frame_id = int(frame_id)
        sample['frame_id'] = frame_id

        depth = io.imread('{}.depth.png'.format(frame))
        depth = depth.astype(np.float32)
        depth = depth / 1000.
        # depth[depth == np.max(depth)] = 0.

        step_x = depth.shape[0] / self.resolution[0]
        step_y = depth.shape[1] / self.resolution[1]

        index_y = [int(step_y * i) for i in
                   range(0, int(depth.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in
                   range(0, int(depth.shape[0] / step_x))]

        depth = depth[:, index_y]
        depth = depth[index_x, :]

        mask = copy(depth)
        mask[mask == np.max(depth)] = 0
        mask[mask != 0] = 1
        original_mask = copy(mask)
        sample['original_mask'] = copy(mask)
        gradient_mask = binary_dilation(mask, iterations=5)
        mask = binary_dilation(mask, iterations=8)
        sample['mask'] = mask
        sample['gradient_mask'] = gradient_mask

        depth[mask == 0] = 0

        sample['depth'] = depth
        sample['noisy_depth'] = add_kinect_noise(copy(depth),
                                                 sigma_fraction=self.noise_scale)
        sample['noisy_depth_octnetfusion'] = add_depth_noise(copy(depth),
                                                             noise_sigma=self.noise_scale,
                                                             seed=42)
        sample['outlier_depth'] = add_outliers(
            copy(sample['noisy_depth_octnetfusion']),
            scale=self.outlier_scale,
            fraction=self.outlier_fraction)

        #sample['sparse_depth'] = add_sparse_depth(copy(depth), percentage=0.01)
        sample['outlier_blob_depth'] = add_outlier_blobs(copy(sample['noisy_depth_octnetfusion']),
                                                         scale=self.outlier_scale,
                                                         fraction=self.outlier_fraction)

        intrinsics = np.loadtxt('{}.intrinsics.txt'.format(frame))
        # adapt intrinsics to camera resolution
        scaling = np.eye(3)
        scaling[1, 1] = self.yscale
        scaling[0, 0] = self.xscale

        sample['intrinsics'] = np.dot(scaling, intrinsics)

        extrinsics = np.loadtxt('{}.extrinsics.txt'.format(frame))
        extrinsics = np.linalg.inv(extrinsics)
        sample['extrinsics'] = extrinsics

        sample['scene_id'] = scene_id

        for key in sample.keys():
            if type(sample[key]) is not np.ndarray and type(
                    sample[key]) is not str:
                sample[key] = np.asarray(sample[key])

        if self.transform:
            sample = self.transform(sample)

        return sample
    @property
    def scenes(self):
        return self._scenes

    def get_grid(self, scene):

        sc, obj = scene.split('/')
        if self.grid_resolution == 256:
            filepath = os.path.join(self.root_dir, sc, self.split, obj,
                                    'voxels', '*.binvox')
        else:
            filepath = os.path.join(self.root_dir, sc, self.split, obj,
                                    'voxels', '*.{}.binvox'.format(
                    self.grid_resolution))

        filepath = glob.glob(filepath)[0]

        # filepath = os.path.join(self.root_dir, 'example', 'voxels', 'chair_0256.binvox')

        with open(filepath, 'rb') as file:
            volume = read_as_3d_array(file)

        array = volume.data.astype(np.int)

        # clean occupancy grids from artifacts
        labels_out = cc3d.connected_components(array)  # 26-connected
        N = np.max(labels_out)
        max_label = 0
        max_label_count = 0
        for segid in range(1, N + 1):
            extracted_image = labels_out * (labels_out == segid)
            extracted_image[extracted_image != 0] = 1
            label_count = np.sum(extracted_image)
            if label_count > max_label_count:
                max_label = segid
                max_label_count = label_count
        array[labels_out != max_label] = 0.

        resolution = 1. / self.grid_resolution

        grid = Voxelgrid(resolution)
        bbox = np.zeros((3, 2))
        bbox[:, 0] = volume.translate
        bbox[:, 1] = bbox[:, 0] + resolution * volume.dims[0]

        grid.from_array(array, bbox)

        return grid
