import torch


class Integrator(torch.nn.Module):

    def __init__(self):

        super(Integrator, self).__init__()

    def forward(self, values, indices, weights, values_volume, weights_volume):

        xs, ys, zs = values_volume.shape

        # reshape tensors
        n1, n2, n3 = values.shape

        values = values.contiguous().view(n1 * n2 * n3, 1)
        values = values.repeat(1, 8)
        indices = indices.contiguous().view(n1 * n2 * n3, 8, 3).long()
        weights = weights.contiguous().view(n1 * n2 * n3, 8)

        n1, n2, n3 = indices.shape
        indices = indices.contiguous().view(n1 * n2, n3).long()
        weights = weights.contiguous().view(n1 * n2, 1).double()
        values = values.contiguous().view(n1 * n2, 1).double()

        valid = get_index_mask(indices, values_volume.shape)

        values = torch.masked_select(values[:, 0], valid)
        indices = extract_indices(indices, mask=valid)
        weights = torch.masked_select(weights[:, 0], valid)

        update = weights * values

        wcache = torch.zeros(values_volume.shape).double().view(xs * ys * zs)
        vcache = torch.zeros(values_volume.shape).double().view(xs * ys * zs)

        index = ys * zs * indices[:, 0] + zs * indices[:, 1] + indices[:, 2]

        wcache.index_add_(0, index, weights)
        vcache.index_add_(0, index, update)

        wcache = wcache.view(xs, ys, zs)
        vcache = vcache.view(xs, ys, zs)

        update = extract_values(indices, vcache)
        weights = extract_values(indices, wcache)

        values_old = extract_values(indices, values_volume)
        weights_old = extract_values(indices, weights_volume)

        value_update = (weights_old * values_old + update) / (weights_old + weights)
        # value_update = update

        # print('update', np.unique(update))
        # print('value update', np.unique(value_update))

        weight_update = weights_old + weights

        insert_values(value_update, indices, values_volume)
        insert_values(weight_update, indices, weights_volume)

        return values_volume, weights_volume


def get_index_mask(indices, shape):
    """
    method to check whether indices are valid
    :param indices: indices to check
    :param shape: constraints for indices
    :return: mask
    """
    xs, ys, zs = shape

    valid = ((indices[:, 0] >= 0) &
             (indices[:, 0] < xs) &
             (indices[:, 1] >= 0) &
             (indices[:, 1] < ys) &
             (indices[:, 2] >= 0) &
             (indices[:, 2] < zs))

    return valid




def extract_values(indices, volume, mask=None):
    """
    method to extract values from volume given indices
    :param indices: positions to extract
    :param volume: volume to extract from
    :param mask: optional mask for extraction
    :return: extracted values
    """

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
    """
    method to extract indices according to mask
    :param indices:
    :param mask:
    :return:
    """

    x = torch.masked_select(indices[:, 0], mask)
    y = torch.masked_select(indices[:, 1], mask)
    z = torch.masked_select(indices[:, 2], mask)

    masked_indices = torch.cat((x.unsqueeze_(1),
                                y.unsqueeze_(1),
                                z.unsqueeze_(1)), dim=1)
    return masked_indices


def insert_values(values, indices, volume):
    """
    method to insert values back into volume
    :param values:
    :param indices:
    :param volume:
    :return:
    """
    volume[indices[:, 0], indices[:, 1], indices[:, 2]] = values