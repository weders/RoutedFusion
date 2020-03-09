import torch
from torch import nn
from torch.nn.functional import normalize


class Extractor(nn.Module):
    '''
    This module extracts voxel rays or blocks around voxels that are given by the
    unprojected 2D depth map as well as the given groundtruth volume and the
    current state of the reconstruction volume
    '''

    def __init__(self, config):

        super(Extractor, self).__init__()

        self.n_points = config.n_points

    def forward(self, depth, extrinsics, intrinsics,
                volume, origin, resolution, weights=None):
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

        intrinsics = intrinsics.double()
        extrinsics = extrinsics.double()

        if torch.cuda.is_available():
            intrinsics = intrinsics.cuda()
            extrinsics = extrinsics.cuda()

            volume = volume.cuda()
            if weights is not None:
                weights = weights.cuda()
            origin = origin.cuda()

        # TODO: this is not valid if two different scenes are in a batch
        xs = volume.shape[0]
        ys = volume.shape[1]
        zs = volume.shape[2]

        depth = depth.double()

        b, h, w = depth.shape

        coords = self.compute_coordinates(depth, extrinsics, intrinsics, origin, resolution)

        # compute rays
        eye_w = extrinsics[:, :3, 3]

        ray_pts, ray_dists = self.extract_values(coords, eye_w, origin, resolution, n_points=int((self.n_points - 1)/2))

        values_current, values_gt, indices, weights, centers, fusion_weights, values_fused = trilinear_interpolation(ray_pts, volume, weights)

        n1, n2, n3 = values_gt.shape

        indices = indices.view(n1, n2, n3, 8, 3)
        weights = weights.view(n1, n2, n3, 8)


        # packing
        values = dict(current=values_current,
                      gt=values_gt,
                      points=ray_pts,
                      depth=depth.view(b, h*w),
                      indices=indices,
                      weights=weights,
                      centers=centers,
                      pcl=coords,
                      fusion_weights=fusion_weights,
                      values_fused=values_fused)

        del volume, extrinsics, intrinsics, origin, weights

        return values

    def compute_coordinates(self, depth, extrinsics, intrinsics, origin, resolution):

        b, h, w = depth.shape
        n_points = h*w

        # generate frame meshgrid
        xx, yy = torch.meshgrid([torch.arange(h, dtype=torch.double), torch.arange(w, dtype=torch.double)])

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
        intrinsics_inv = intrinsics.inverse()

        homogenuous = torch.ones((b, 1, n_points)).double()

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

        if ellipsoid:
            xp1 = torch.ones_like(direction)[:, :, 0]
            yp1 = torch.ones_like(direction)[:, :, 0]
            zp1 = (-torch.mul(xp1, direction[:, :, 0]) - torch.mul(yp1, direction[:, :, 1]))/direction[:, :, 2]

            perp1 = torch.stack((xp1, yp1, zp1), dim=2)
            perp1 = normalize(perp1, p=2, dim=2)
            perp2 = torch.cross(direction, perp1)
            perp2 = normalize(perp2, p=2, dim=2)

            for i in range(1, 2):

                p1 = points[0] + i*bin_size*perp1
                p1N = points[0] - i*bin_size*perp1

                p2 = points[0] + i * bin_size * perp2
                p2N = points[0] - i * bin_size * perp2

                ellip.append(p1.clone())
                ellip.append(p1N.clone())
                ellip.append(p2.clone())
                ellip.append(p2N.clone())

        for i in range(1, n_points+1):
            point = center_v + i*bin_size*direction
            pointN = center_v - i*bin_size*direction
            points.append(point.clone())
            points.insert(0, pointN.clone())

            if i <= 1 and ellipsoid:
                for j in range(1, 2):
                    p1 = point + j * bin_size * perp1
                    p1N = pointN - j * bin_size * perp1

                    p2 = point + j * bin_size * perp2
                    p2N = pointN - j * bin_size * perp2

                    ellip.append(p1.clone())
                    ellip.append(p1N.clone())
                    ellip.append(p2.clone())
                    ellip.append(p2N.clone())


            dist = i*bin_size*torch.ones_like(point)[:, :, 0]
            distN = -1.*dist

            dists.append(dist)
            dists.insert(0, distN)

        if ellipsoid:
            points = points + ellip

        dists = torch.stack(dists, dim=2)
        points = torch.stack(points, dim=2)

        return points, dists


def interpolation_weights(points, mode='center'):

    if mode == 'center':
        # compute step direction
        center = 0.5 * torch.ones_like(points) + torch.floor(points)
        neighbor = torch.sign(center - points)
    else:
        center = torch.floor(points)
        neighbor = torch.ones_like(points)

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


def trilinear_interpolation(points, volume, volume_gt, fusion_weights_volume):

    b, h, n, dim = points.shape

    def analyze_center(points):

        centers = torch.floor(points[0, :, 4, :]).long()
        valid = get_index_mask(centers, volume.shape)
        valid_idx = torch.nonzero(valid)[:, 0]
        values_valid = extract_values(centers, volume_gt, valid)

        values = torch.zeros_like(centers[:, 0]).double()
        values[valid_idx] = values_valid


        return values

    center_values = analyze_center(points)

    weights, indices = interpolation_weights(points, mode='center')

    n1, n2, n3 = indices.shape

    indices = indices.contiguous().view(n1*n2, n3).long()

    valid = get_index_mask(indices, volume.shape)

    valid_idx = torch.nonzero(valid)[:, 0]

    vv = extract_values(indices, volume, valid)
    vv_gt = extract_values(indices, volume_gt, valid)
    f_weights = extract_values(indices, fusion_weights_volume, valid)

    values = torch.zeros_like(valid).double()
    values_gt = torch.zeros_like(valid).double()
    f_weights_values = torch.zeros_like(valid).double()

    values[valid_idx] = vv
    values_gt[valid_idx] = vv_gt
    f_weights_values[valid_idx] = f_weights

    values = values.view(weights.shape)
    values_gt = values_gt.view(weights.shape)
    f_weights_values = f_weights_values.view(weights.shape)

    values = values * weights #* f_weights_values
    values_fused = values * weights * f_weights_values
    values_gt = values_gt*weights

    values = torch.sum(values, dim=1)
    values_gt = torch.sum(values_gt, dim=1)
    values_fused = torch.sum(values_fused, dim=1)
    fusion_weights = torch.sum(f_weights_values * weights, dim=1)

    values = values.view(b, h, n)
    values_gt = values_gt.view(b, h, n)
    values_fused = values_fused.view(b, h, n)
    fusion_weights = fusion_weights.view(b, h, n)

    indices = indices.view(n1, n2, n3)

    # center_points = values_gt[0, :, 4].detach().numpy()
    # for point in center_points:
    #     print(point)

    del vv, vv_gt, f_weights, f_weights_values, valid, valid_idx

    return values.float(), values_gt.float(), indices, weights, center_values, fusion_weights.float(), values_fused.float()



