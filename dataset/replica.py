import os
import glob
import pyquaternion

import numpy as np
import pywavefront

from skimage import io, transform
from torch.utils.data import Dataset
from copy import copy

from graphics import Voxelgrid

import h5py

from time import sleep
from weight_learning.utils import add_axial_noise, \
    add_random_zeros, add_lateral_noise, add_outliers, add_kinect_noise
from datasets.utils import read_dat_groundtruth, \
    read_projections, read_camera_file


class Replica(Dataset):

    def __init__(self, root_dir, scene='apartment_0', frame_list=None, resolution=(240, 320), transform=None):

        self.root_dir = root_dir
        self.scene = scene

        self.frame_list = frame_list

        self._load_color()
        self._load_depth()
        self._load_cameras()

        self.resolution = resolution
        self.xscale = resolution[0]/480
        self.yscale = resolution[1]/640

        self.transform = transform

    def _load_depth(self):

        if self.frame_list is None:
            self.depth_images = glob.glob(os.path.join(self.root_dir, 'depth', '*.png'))

        else:
            # reading files from list
            self.depth_images = []
            with open(os.path.join(self.root_dir, self.frame_list)) as file:
                for line in file:
                    self.depth_images.append(os.path.join(self.root_dir, 'depth', line.rstrip()))

        self.depth_images = sorted(self.depth_images, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))

    def _load_color(self):

        if self.frame_list is None:
            self.color_images = glob.glob(os.path.join(self.root_dir, 'images', '*.png'))

        else:

            self.color_images = []
            with open(os.path.join(self.root_dir, self.frame_list)) as file:
                for line in file:
                    self.color_images.append(os.path.join(self.root_dir, 'images', line.rstrip()))

        self.color_images = sorted(self.color_images, key=lambda x: int(os.path.splitext(x.split('/')[-1])[0]))

    def _load_cameras(self):

        self.cameras = []

        with open(os.path.join(self.root_dir, 'cameras.txt')) as file:

            for line in file:
                elems = line.rstrip().split(' ')
                mat = []
                for p in elems:
                    if p == '':
                        continue
                    mat.append(float(p))

                position = np.asarray(mat[:3])
                rotation = np.asarray(mat[3:])

                M = np.eye(4)
                M[0, 0] = -1.
                M[1, 1] = 1.
                M[2, 2] = 1.

                qw = rotation[0]
                qx = rotation[1]
                qy = rotation[2]
                qz = rotation[3]

                quaternion = pyquaternion.Quaternion(qw, qx, qy, qz)
                rotation = quaternion.rotation_matrix

                extrinsics = np.eye(4)
                extrinsics[:3, :3] = rotation
                extrinsics[:3, 3] = position

                extrinsics = np.dot(M, extrinsics)


                extrinsics[:3, 2] *= -1.

                # extrinsics = np.linalg.inv(extrinsics)

                self.cameras.append(np.copy(extrinsics))

        # selecting only cameras in list
        if self.frame_list is not None:

            cameras_filtered = []

            for frame in self.color_images:
                i = int(frame.split('/')[-1].split('.')[0])
                cameras_filtered.append(self.cameras[i])

            self.cameras = cameras_filtered

    def __len__(self):
        return len(self.color_images)

    def __getitem__(self, item):

        sample = dict()

        sample['frame_id'] = item

        # load image
        file = self.color_images[item]
        image = io.imread(file)
        image = image[:, :, :3]
        image = transform.resize(image, self.resolution)
        sample['image'] = np.asarray(image)

        # load depth map
        file = self.depth_images[item]
        depth = io.imread(file).astype(np.float32)

        step_x = depth.shape[0] / self.resolution[0]
        step_y = depth.shape[1] / self.resolution[1]

        index_y = [int(step_y * i) for i in
                   range(0, int(depth.shape[1] / step_y))]
        index_x = [int(step_x * i) for i in
                   range(0, int(depth.shape[0] / step_x))]

        depth = depth[:, index_y]
        depth = depth[index_x, :]

        depth /= 1000.

        sample['depth'] = np.asarray(depth)

        # load extrinsics
        extrinsics = self.cameras[item]
        sample['extrinsics'] = extrinsics

        hfov = 90.
        f = 640./2.*(1./np.tan(np.deg2rad(hfov)/2))

        # load intrinsics
        intrinsics = np.asarray([[f, 0., 320.],
                                 [0., f, 240.],
                                 [0., 0., 1.]])

        scaling = np.eye(3)
        scaling[0, 0] = self.xscale
        scaling[1, 1] = self.yscale
        sample['intrinsics'] = np.dot(scaling, intrinsics)

        sample['scene_id'] = self.scene

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_grid(self):

        bbox_file = os.path.join(self.root_dir, 'bbox.txt')
        grid_file = os.path.join(self.root_dir, '{}_occ.hf5'.format(self.scene))

        self._bbox = np.loadtxt(bbox_file)
        self._origin = self._bbox[:, 0]

        with h5py.File(grid_file, 'r') as hf:
            volume = hf['TSDF'][:]

        voxelgrid = Voxelgrid(0.02)
        voxelgrid.from_array(volume, self._bbox)

        return voxelgrid


if __name__ == '__main__':

    from tqdm import tqdm
    from mayavi import mlab

    import matplotlib.pyplot as plt

    dataset = Replica('/local/home/data/001-real-time-depth-map-fusion/replica/apartment_0/habitat/')

    def pixel_to_camera_coord(point, intrinsics, z):

        camera_coord = np.zeros(3, )

        camera_coord[2] = z
        camera_coord[1] = z * (point[1] - intrinsics[1, 2]) / intrinsics[1, 1]
        camera_coord[0] = z * (point[0] - intrinsics[0, 1] * camera_coord[1] - intrinsics[0, 2]) / intrinsics[0, 0]

        return camera_coord

    frame_counter = 0
    pointcloud = []

    # frames = np.random.choice(np.arange(0, len(dataset), 1), 20)
    frames = np.arange(0, len(dataset), 1)

    for f in tqdm(frames, total=len(frames)):

        frame = dataset[f]
        depth = frame['depth']
        # depth = np.flip(depth, axis=0)
        # plt.imshow(depth)
        # plt.show()

        for i in range(0, depth.shape[0]):
            for j in range(0, depth.shape[1]):

                z = depth[i, j]

                p = np.asarray([j, i, z])
                c = pixel_to_camera_coord([j, i], frame['intrinsics'], z)
                c = np.concatenate([c, np.asarray([1.])])
                w = np.dot(frame['extrinsics'], c)

                pointcloud.append(w)

        frame_counter += 1

        # if (frame_counter + 1) % 5 == 0:
        #     print(frame_counter)
        #     array = np.asarray(pointcloud)
        #     print(np.max(array, axis=0))
        #     print(np.min(array, axis=0))
        #
        #     mlab.points3d(array[:, 0],
        #                   array[:, 1],
        #                   array[:, 2],
        #                   scale_factor=0.05)
        #
        #     mlab.show()
        #     mlab.close(all=True)

    array = np.asarray(pointcloud)
    print(np.max(array, axis=0))
    print(np.min(array, axis=0))

    # array = np.asarray(pointcloud)
    # mlab.points3d(array[:, 0],
    #               array[:, 1],
    #               array[:, 2],
    #               scale_factor=0.05)
    #
    # mlab.show()
