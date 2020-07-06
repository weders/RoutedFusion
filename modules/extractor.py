import torch
import datetime
from torch import nn
from torch.nn.functional import normalize


class Extractor(nn.Module):
    '''
    This module extracts voxel rays or blocks around voxels that are given by the
    reconstructed 2D depth map as well as the given groundtruth volume and the
    current state of the reconstruction volume
    '''

    def __init__(self, config):

        super(Extractor, self).__init__()

        self.config = config

        self.n_points = 9
        self.mode = 'ray'

    def forward(self, depth, extrinsics, intrinsics, tsdf_volume, origin, resolution, weights_volume=None):
        '''
        Computes the forward pass of extracting the rays/blocks and the corresponding coordinates

        :param depth: depth map with the values that define the center voxel of the ray/block
        :param extrinsics: camera extrinsics matrix for mapping
        :param intrinsics: camera intrinsics matrix for mapping
        :param volume_gt: groundtruth voxel volume
        :param volume_current: current state of reconstruction volume
        :param origin: origin of groundtruth volume in world coordinates
        :param resolution: resolution of voxel volume
        :return: values/voxels of groundtruth and current as well as at its coordinates and indices
        '''

        tsdf_volume = tsdf_volume

        device = depth.get_device()

        intrinsics = intrinsics.float()
        extrinsics = extrinsics.float()

        if device >= 0:
            intrinsics = intrinsics.to(device)
            extrinsics = extrinsics.to(device)

            tsdf_volume = tsdf_volume.to(device)
            weights_volume = weights_volume.to(device)
            origin = origin.to(device)

        b, h, w = depth.shape

        self.depth = depth.contiguous().view(b, h*w)

        coords = self.compute_coordinates(depth, extrinsics, intrinsics, origin, resolution)

        # compute rays
        eye_w = extrinsics[:, :3, 3]

        ray_pts, ray_dists = self.extract_values(coords, eye_w, origin, resolution, n_points=int((self.n_points - 1)/2))

        fusion_values, indices, weights, fusion_weights = trilinear_interpolation(ray_pts, tsdf_volume, weights_volume)

        n1, n2, n3 = fusion_values.shape

        indices = indices.view(n1, n2, n3, 8, 3)
        weights = weights.view(n1, n2, n3, 8)

        # packing
        values = dict(fusion_values=fusion_values,
                      fusion_weights=fusion_weights,
                      points=ray_pts,
                      depth=depth.view(b, h*w),
                      indices=indices,
                      weights=weights,
                      pcl=coords)

        del extrinsics, intrinsics, origin, weights_volume, tsdf_volume

        return values


    def compute_coordinates(self, depth, extrinsics, intrinsics, origin, resolution):

        b, h, w = depth.shape
        n_points = h*w

        # generate frame meshgrid
        xx, yy = torch.meshgrid([torch.arange(h, dtype=torch.float),
                                 torch.arange(w, dtype=torch.float)])

        if torch.cuda.is_available():
            xx = xx.cuda()
            yy = yy.cuda()

        # flatten grid coordinates and bring them to batch size
        xx = xx.contiguous().view(1, h*w, 1).repeat((b, 1, 1))
        yy = yy.contiguous().view(1, h*w, 1).repeat((b, 1, 1))
        zz = depth.contiguous().view(b, h*w, 1)

        # generate points in pixel space
        points_p = torch.cat((yy, xx, zz), dim=2).clone()

        # invert
        intrinsics_inv = intrinsics.inverse().float()

        homogenuous = torch.ones((b, 1, n_points))

        if torch.cuda.is_available():
            homogenuous = homogenuous.cuda()

        # transform points from pixel space to camera space to world space (p->c->w)
        points_p[:, :, 0] *= zz[:, :, 0]
        points_p[:, :, 1] *= zz[:, :, 0]
        points_c = torch.matmul(intrinsics_inv, torch.transpose(points_p, dim0=1, dim1=2))
        points_c = torch.cat((points_c, homogenuous), dim=1)
        points_w = torch.matmul(extrinsics[:3], points_c)
        points_w = torch.transpose(points_w, dim0=1, dim1=2)[:, :, :3]

        del xx, yy, homogenuous, points_p, points_c, intrinsics_inv
        return points_w

    def extract_values(self, coords, eye, origin, resolution,
                       bin_size=1.0, n_points=4):


        center_v = (coords - origin) / resolution
        eye_v = (eye - origin)/resolution

        direction = center_v - eye_v
        direction = normalize(direction, p=2, dim=2)

        points = [center_v]

        ellip = []

        dist = torch.zeros_like(center_v)[:, :, 0]
        dists = [dist]

        for i in range(1, n_points+1):
            point = center_v + i*bin_size*direction
            pointN = center_v - i*bin_size*direction
            points.append(point.clone())
            points.insert(0, pointN.clone())

            dist = i*bin_size*torch.ones_like(point)[:, :, 0]
            distN = -1.*dist

            dists.append(dist)
            dists.insert(0, distN)

        dists = torch.stack(dists, dim=2)
        points = torch.stack(points, dim=2)

        return points, dists


def interpolate(points):
    """
    Method to compute the interpolation indices and weights for trilinear
    interpolation
    """

    n = points.shape[0]

    # get indices
    indices = torch.floor(points)

    # compute interpolation distance
    df = torch.abs(points - indices)

    # get interpolation indices
    xx, yy, zz = torch.meshgrid([torch.arange(0, 2),
                                 torch.arange(0, 2),
                                 torch.arange(0, 2)])

    xx = xx.contiguous().view(8)
    yy = yy.contiguous().view(8)
    zz = zz.contiguous().view(8)

    shift = torch.stack([xx, yy, zz], dim=1)

    if points.get_device() >= 0:
        shift = shift.to(points.get_device())

    # reshape
    shift = shift.unsqueeze_(0)
    indices = indices.unsqueeze_(1)

    # compute indices
    indices = indices + shift

    # init weights
    weights = torch.zeros_like(indices).sum(dim=-1)

    # compute weights
    weights[:, 0] = (1 - df[:, 0]) * (1 - df[:, 1]) * (1 - df[:, 2])
    weights[:, 1] = (1 - df[:, 0]) * (1 - df[:, 1]) * df[:, 2]
    weights[:, 2] = (1 - df[:, 0]) * df[:, 1] * (1 - df[:, 2])
    weights[:, 3] = (1 - df[:, 0]) * df[:, 1] * df[:, 2]
    weights[:, 4] = df[:, 0] * (1 - df[:, 1]) * (1 - df[:, 2])
    weights[:, 5] = df[:, 0] * (1 - df[:, 1]) * df[:, 2]
    weights[:, 6] = df[:, 0] * df[:, 1] * (1 - df[:, 2])
    weights[:, 7] = df[:, 0] * df[:, 1] * df[:, 2]

    weights = weights.unsqueeze_(-1)

    return indices, weights


def interpolation_weights(points, mode='center'):

    if mode == 'center':
        # compute step direction
        center = torch.floor(points) + 0.5 * torch.ones_like(points)
        neighbor = torch.sign(center - points)
    else:
        center = torch.floor(points)
        neighbor = torch.sign(center - points)

    # index of center voxel
    idx = torch.floor(points)

    # reshape for pytorch compatibility
    b, h, n, dim = idx.shape
    points = points.contiguous().view(b * h * n, dim)
    center = center.contiguous().view(b * h * n, dim)
    idx = idx.view(b * h * n, dim)
    neighbor = neighbor.view(b * h * n, dim)

    # center x.0
    alpha = torch.abs(points - center)  # always positive
    alpha_inv = 1 - alpha

    weights = []
    indices = []

    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                if i == 0:
                    w1 = alpha_inv[:, 0]
                    ix = idx[:, 0]
                else:
                    w1 = alpha[:, 0]
                    ix = idx[:, 0] + neighbor[:, 0]
                if j == 0:
                    w2 = alpha_inv[:, 1]
                    iy = idx[:, 1]
                else:
                    w2 = alpha[:, 1]
                    iy = idx[:, 1] + neighbor[:, 1]
                if k == 0:
                    w3 = alpha_inv[:, 2]
                    iz = idx[:, 2]
                else:
                    w3 = alpha[:, 2]
                    iz = idx[:, 2] + neighbor[:, 2]

                weights.append((w1 * w2 * w3).unsqueeze_(1))
                indices.append(torch.cat((ix.unsqueeze_(1),
                                          iy.unsqueeze_(1),
                                          iz.unsqueeze_(1)),
                                         dim=1).unsqueeze_(1))

    weights = torch.cat(weights, dim=1)
    indices = torch.cat(indices, dim=1)

    del points, center, idx, neighbor, alpha, alpha_inv, ix, iy, iz, w1, w2, w3

    return weights, indices


def get_index_mask(indices, shape):

    xs, ys, zs = shape

    valid = ((indices[:, 0] >= 0) &
             (indices[:, 0] < xs) &
             (indices[:, 1] >= 0) &
             (indices[:, 1] < ys) &
             (indices[:, 2] >= 0) &
             (indices[:, 2] < zs))

    return valid


def extract_values(indices, volume, mask=None, fusion_weights=None):

    if mask is not None:
        x = torch.masked_select(indices[:, 0], mask)
        y = torch.masked_select(indices[:, 1], mask)
        z = torch.masked_select(indices[:, 2], mask)
    else:
        x = indices[:, 0]
        y = indices[:, 1]
        z = indices[:, 2]

    return volume[x, y, z]


def extract_indices(indices, mask):

    x = torch.masked_select(indices[:, 0], mask)
    y = torch.masked_select(indices[:, 1], mask)
    z = torch.masked_select(indices[:, 2], mask)

    masked_indices = torch.cat((x.unsqueeze_(1),
                                y.unsqueeze_(1),
                                z.unsqueeze_(1)), dim=1)
    return masked_indices


def insert_values(values, indices, volume):
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = values


def trilinear_interpolation(points, tsdf_volume, weights_volume):

    b, h, n, dim = points.shape

    #get interpolation weights
    weights, indices = interpolation_weights(points)

    # points = points.view(b * h * n, dim)
    # indices, weights = interpolate(points)

    n1, n2, n3 = indices.shape
    indices = indices.contiguous().view(n1*n2, n3).long()

    #TODO: change to replication padding instead of zero padding
    #TODO: double check indices

    # get valid indices
    valid = get_index_mask(indices, tsdf_volume.shape)
    valid_idx = torch.nonzero(valid)[:, 0]

    tsdf_values = extract_values(indices, tsdf_volume, valid)
    tsdf_weights = extract_values(indices, weights_volume, valid)

    value_container = torch.zeros_like(valid).double()
    weight_container = torch.zeros_like(valid).double()

    value_container[valid_idx] = tsdf_values
    weight_container[valid_idx] = tsdf_weights

    value_container = value_container.view(weights.shape)
    weight_container = weight_container.view(weights.shape)

    # trilinear interpolation
    fusion_values = torch.sum(value_container * weights, dim=1)
    fusion_weights = torch.sum(weight_container * weights, dim=1)

    fusion_values = fusion_values.view(b, h, n)
    fusion_weights = fusion_weights.view(b, h, n)

    indices = indices.view(n1, n2, n3)

    return fusion_values.float(), indices, weights, fusion_weights.float()



