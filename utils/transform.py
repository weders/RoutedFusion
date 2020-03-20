import torch
import numpy as np


class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        result = {}

        for key in sample.keys():
            if type(sample[key]) is np.ndarray:

                if key == 'image':
                    # swap color axis because
                    # numpy image: H x W x C
                    # torch image: C X H X W
                    image = sample[key].transpose((2, 0, 1))
                    image = torch.from_numpy(image)
                    result[key] = image
                    continue

                result[key] = torch.from_numpy(sample[key])

            else:
                result[key] = sample[key]

        return result


def to_device(data, device):
    for k in data.keys():
        if torch.is_tensor(data[k]):
            data[k] = data[k].to(device)
    return data
