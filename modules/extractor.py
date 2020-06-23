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

        intrinsics = intrinsics.float()
        extrinsics = extrinsics.float()

        if torch.cuda.is_available():
            intrinsics = intrinsics.cuda()
            extrinsics = extrinsics.cuda()

            tsdf_volume = tsdf_volume.cuda()
            weights_volume = weights_volume.cuda()
            origin = origin.cuda()

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



    def extract_fusion_weights(self, coords, eye, origin, resolution, volume):

        centers = (coords - origin) / resolution
        indices = torch.floor(centers)

        n1, n2, n3 = indices.shape

        indices = indices.contiguous().view(n1 * n2, n3).long()

        valid = get_index_mask(indices, volume.shape)
        valid_idx = torch.nonzero(valid)[:, 0]

        fusion_weights = -1.*torch.ones_like(valid).double()

        extracted_fusion_weights = extract_values(indices, volume, valid)

        fusion_weights[valid_idx] = extracted_fusion_weights

        fusion_weights = fusion_weights.unsqueeze_(0)
        fusion_weights = fusion_weights.unsqueeze_(-1)

        return fusion_weights.float()

    @staticmethod
    def extract_cubes(coords, size=(5, 5, 5)):

        # generate block grid
        xx, yy, zz = torch.meshgrid([torch.arange(-2, 3, dtype=torch.int),
                                     torch.arange(-2, 3, dtype=torch.int),
                                     torch.arange(-2, 3, dtype=torch.int)])

        xx = xx.contiguous().view(size[0]*size[1]*size[2], 1)
        yy = yy.contiguous().view(size[0]*size[1]*size[2], 1)
        zz = zz.contiguous().view(size[0]*size[1]*size[2], 1)

        block = torch.cat((xx, yy, zz), dim=1)
        block_coords = block + coords[:, :, None]
        return block_coords

    @staticmethod
    def extract_rays(coords, eye, origin, resolution, gpu=True):

        center_v = (coords - origin)/resolution

        eye_v = (eye - origin)/resolution

        current_idx = torch.floor(center_v)
        current_idxN = torch.floor(center_v)

        # tv = torch.min(center_v - current_idx, dim=2)[0].unsqueeze_(-1)
        # res = (center_v - current_idx) - tv
        # center_v = current_idx + res

        current_point = center_v.clone()
        current_pointN = center_v.clone()

        delta = center_v - eye_v
        delta = normalize(delta, dim=2)
        delta = 1./delta


        direction = center_v - eye_v
        direction = normalize(direction, dim=2)

        if torch.cuda.is_available() and gpu:
            delta = delta.cuda()

        step = torch.sign(delta)

        ones = torch.ones(current_idx.shape).double()
        if torch.cuda.is_available() and gpu:
            ones = ones.cuda()

        next = torch.where(step > 0,
                           delta*(current_idx + ones - center_v),
                           delta*(current_idx - center_v))
        nextN = torch.where(step < 0,
                            delta*(center_v - current_idx - ones),
                            delta*(center_v - current_idx))

        tDelta = delta
        tDelta = torch.mul(step, tDelta)

        rays = [current_idx.clone()]

        cum_dist = torch.zeros((next.shape[0], next.shape[1])).double()
        cum_distN = torch.zeros((next.shape[0], next.shape[1])).double()

        if torch.cuda.is_available() and gpu:
            cum_dist = cum_dist.cuda()
            cum_distN = cum_distN.cuda()

        dists = [cum_dist.clone()]
        pts = [current_point.clone()]

        for i in range(0, 15):

            side = torch.argmin(next, dim=2)
            sideN = torch.argmin(nextN, dim=2)

            no_update = torch.zeros(next[:, :, 0].shape).double()

            if torch.cuda.is_available() and gpu:
                no_update = no_update.cuda()

            for j in range(0, 3):

                epsilon = 10.e-4

                cache = torch.zeros_like(next)
                cacheN = torch.zeros_like(nextN)

                cache[:, :, 0] = torch.where(direction[:, :, 0] < 0,
                                             (direction[:, :, 0]) * next[:, :, j] - epsilon,
                                             direction[:, :, 0]*next[:, :, j])
                cache[:, :, 1] = torch.where(direction[:, :, 1] < 0,
                                             (direction[:, :, 1]) * next[:, :, j] - epsilon,
                                             direction[:, :, 1]*next[:, :, j])
                cache[:, :, 2] = torch.where(direction[:, :, 2] < 0,
                                             (direction[:, :, 2]) * next[:, :, j] - epsilon,
                                             direction[:, :, 2]*next[:, :, j])

                cacheN[:, :, 0] = torch.where(direction[:, :, 0] > 0,
                                              (-1.*direction[:, :, 0]) * nextN[:, :, j] - epsilon,
                                              -1.*direction[:, :, 0]*nextN[:, :, j])
                cacheN[:, :, 1] = torch.where(direction[:, :, 1] > 0,
                                              (-1.*direction[:, :, 1]) * nextN[:, :, j] - epsilon,
                                              -1.*direction[:, :, 1]*nextN[:, :, j])
                cacheN[:, :, 2] = torch.where(direction[:, :, 2] > 0,
                                              (-1.*direction[:, :, 2]) * nextN[:, :, j] - epsilon,
                                              -1.*direction[:, :, 2]*nextN[:, :, j])

                current_point = torch.where(torch.stack((side, side, side), dim=2) == j,
                                             cache, current_point)
                current_pointN = torch.where(torch.stack((sideN, sideN, sideN), dim=2) == j,
                                      cacheN, current_pointN)

            current_point += center_v
            current_pointN += center_v

            cum_dist[:, :] = torch.where(side == 0, next[:, :, 0], no_update).clone()
            cum_dist[:, :] = torch.where(side == 1, next[:, :, 1], cum_dist).clone()
            cum_dist[:, :] = torch.where(side == 2, next[:, :, 2], cum_dist).clone()

            cum_distN[:, :] = -1. * torch.where(sideN == 0, nextN[:, :, 0],
                                                no_update).clone()
            cum_distN[:, :] = -1. * torch.where(sideN == 1, nextN[:, :, 1],
                                                cum_distN).clone()
            cum_distN[:, :] = -1. * torch.where(sideN == 2, nextN[:, :, 2],
                                                cum_distN).clone()

            next[:, :, 0] += torch.where(side == 0, tDelta[:, :, 0], no_update)
            next[:, :, 1] += torch.where(side == 1, tDelta[:, :, 1], no_update)
            next[:, :, 2] += torch.where(side == 2, tDelta[:, :, 2], no_update)

            nextN[:, :, 0] += torch.where(sideN == 0, tDelta[:, :, 0], no_update)
            nextN[:, :, 1] += torch.where(sideN == 1, tDelta[:, :, 1], no_update)
            nextN[:, :, 2] += torch.where(sideN == 2, tDelta[:, :, 2], no_update)

            current_idx[:, :, 0] += torch.where(side == 0, step[:, :, 0], no_update)
            current_idx[:, :, 1] += torch.where(side == 1, step[:, :, 1], no_update)
            current_idx[:, :, 2] += torch.where(side == 2, step[:, :, 2], no_update)

            current_idxN[:, :, 0] += torch.where(sideN == 0, -step[:, :, 0], no_update)
            current_idxN[:, :, 1] += torch.where(sideN == 1, -step[:, :, 1], no_update)
            current_idxN[:, :, 2] += torch.where(sideN == 2, -step[:, :, 2], no_update)

            rays.insert(0, current_idxN.clone())
            rays.append(current_idx.clone())

            dists.insert(0, cum_distN.clone())
            dists.append(cum_dist.clone())

            pts.insert(0, current_pointN.clone())
            pts.append(current_point.clone())

        rays = torch.stack(rays, dim=2)
        dists = torch.stack(dists, dim=2)
        pts = torch.stack(pts, dim=2)

        del delta, cum_dist, cum_distN, ones, no_update, current_idx, current_idxN, current_point, current_pointN
        del next, nextN, cache, cacheN

        return rays, dists, pts

    def extract_values(self, coords, eye, origin, resolution,
                       bin_size=1.0, n_points=4,
                       ellipsoid=False):


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

        if ellipsoid:
            points = points + ellip

        dists = torch.stack(dists, dim=2)
        points = torch.stack(points, dim=2)

        return points, dists

    @staticmethod
    def extract_voxels(rays, volume_gt, volume_current):

        xs, ys, zs = volume_gt.shape

        b, n_ray, l_ray, dim = rays.shape

        voxel_coords = rays.contiguous().view(b*n_ray*l_ray, dim)

        valid = ((voxel_coords[:, 0] >= 0) &
                 (voxel_coords[:, 0] < xs) &
                 (voxel_coords[:, 1] >= 0) &
                 (voxel_coords[:, 1] < ys) &
                 (voxel_coords[:, 2] >= 0) &
                 (voxel_coords[:, 2] < zs))

        valid_idx = valid.nonzero()
        valid_x = valid_idx[:, 0]

        # get coordinates which are valid
        x = torch.masked_select(voxel_coords[:, 0], valid).long()
        y = torch.masked_select(voxel_coords[:, 1], valid).long()
        z = torch.masked_select(voxel_coords[:, 2], valid).long()

        # extract voxel values
        values_gt = volume_gt[x, y, z].float()
        values_current = volume_current[x, y, z].float()

        # generate voxel container
        voxels_gt = torch.zeros((b * n_ray * l_ray, 1))
        voxels_current = torch.zeros((b * n_ray * l_ray, 1))

        if torch.cuda.is_available():
            voxels_gt = voxels_gt.cuda()
            voxels_current = voxels_current.cuda()

        # assign voxel values to container
        voxels_gt[valid_x, 0] = values_gt
        voxels_current[valid_x, 0] = values_current

        voxels_gt = voxels_gt.view(b, n_ray, l_ray)
        voxels_current = voxels_current.view(b, n_ray, l_ray)

        return voxels_current, voxels_gt

    @staticmethod
    def interpolate_values(voxels_current, voxels_gt, dists, limit=.2, n_values=10):

        b, n_ray, l_ray = voxels_gt.shape

        values_gt = torch.zeros(b, n_ray, n_values)
        values_current = torch.zeros(b, n_ray, n_values)
        count = torch.zeros(b, n_ray, n_values)

        interpolation_indices = -1. * torch.ones((b, n_ray, l_ray))
        interpolation_mask = torch.ones(interpolation_indices.shape)

        no_value = torch.zeros(voxels_gt.shape)
        a_value = torch.ones(voxels_gt.shape)

        if torch.cuda.is_available():
            values_gt = values_gt.cuda()
            values_current = values_current.cuda()
            count = count.cuda()

            interpolation_indices = interpolation_indices.cuda()
            interpolation_mask = interpolation_mask.cuda()

            no_value = no_value.cuda()
            a_value = a_value.cuda()

        # buckets definition
        radius = 2*limit
        step = radius/n_values
        n_steps = int(radius/step)

        for i in range(n_steps):

            low = -limit + i * step
            high = -limit + (i + 1) * step

            valid_current = torch.where(((dists > low) & (dists.le(high))), voxels_current, no_value)
            valid_gt = torch.where(((dists > low) & (dists.le(high))), voxels_gt, no_value)
            interpolation_indices = torch.where(((dists >= low) & (dists.le(high))),
                                                i * interpolation_mask,
                                                interpolation_indices)
            nonzeros = torch.where(((dists > low) & (dists.le(high))), a_value, no_value)

            values_current[:, :, i] = torch.sum(valid_current, dim=2)
            values_gt[:, :, i] = torch.sum(valid_gt, dim=2)
            count[:, :, i] = torch.sum(nonzeros, dim=2)

        ones = torch.ones(count.shape)
        if torch.cuda.is_available():
            ones = ones.cuda()

        count = torch.where(count == 0, ones, count)
        values_gt /= count
        values_current /= count

        # free up GPU memory
        del count, interpolation_mask, no_value, a_value, ones, voxels_current, voxels_gt

        return values_current, values_gt, interpolation_indices

    @staticmethod
    def mask_voxel(current, gt, dists, truncation=0.5):

        print(torch.min(dists), torch.max(dists))


        b, h, w = current.shape

        current = current.view(b*h*w)
        gt = gt.view(b*h*w)

        abs_dists = torch.abs(dists).view(b*h*w)

        ones = -1.*torch.ones(current.shape)
        zeros = torch.zeros(current.shape)

        if torch.cuda.is_available():
            ones = ones.cuda()
            zeros = zeros.cuda()

        mask = torch.where(abs_dists <= truncation, zeros, ones)
        mask = mask.view(b, h, w)

        # free up GPU memory
        del ones, zeros

        return mask

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



