import numpy as np
import random
import torch
import time


def add_kinect_noise(depth, sigma_fraction=0.05):

    r = np.random.uniform(0., 1., depth.shape)
    sign = np.ones(depth.shape)
    sign[r < 0.5] = -1.0
    sigma = sigma_fraction*depth
    magnitude = sigma*(1.0 - np.exp(-0.5*np.power(r, 2)))
    depth += sign*magnitude
    depth[depth < 0] = 0.
    return depth


def add_axial_noise(x, std=0.05, depth_dependency=False, radial_dependency=False):

    if radial_dependency is False and depth_dependency is False:

        x += np.random.normal(0, scale=std)
        return x

    if depth_dependency:

        sigma = 0.0012 + 0.0019*np.power((x - 0.4), 2)
        x += np.random.normal(0, scale=sigma)
        return x


def add_random_zeros(x, p=0.9):

    mask = np.random.uniform(0, 1, x.shape)
    mask[mask >= p] = 0.0
    mask[mask > 0.0] = 1.0

    return np.multiply(x, mask)


def add_lateral_noise(x, focal_length=557, method='gaussian'):

    pixels = np.arange(-int(x.shape[1]/2), int(x.shape[1]/2), dtype=np.int32)
    theta = np.arctan(pixels/focal_length)

    sigma_l = 0.8 + 0.035*theta/(np.pi/2. - theta)

    x += np.random.normal(0, scale=sigma_l)
    return x


def add_depth_noise(depthmaps, noise_sigma, seed):

    # add noise
    if noise_sigma > 0:
        random.seed(time.clock())
        np.random.seed(int(time.clock()))
        sigma = noise_sigma
        noise = np.random.normal(0, 1, size=depthmaps.shape).astype(np.float32)
        depthmaps = depthmaps + noise * sigma * depthmaps

    return depthmaps


def add_lateral_and_axial_noise(x, focal_length):

    pixels = np.arange(-int(x.shape[1] / 2), int(x.shape[1] / 2), dtype=np.int32)
    theta = np.arctan(pixels / focal_length)

    sigma = 0.0012 + 0.0019*(x - 0-4)**2 + 0.0001/np.sqrt(x)*(theta**2)/(np.pi/2 - theta)**2

    x += np.random.normal(0, scale=sigma)
    return x


def add_outliers(x, scale=5, fraction=0.99):

    # check for invalid data points
    x[x < 0.] = 0.

    random.seed(time.clock())
    np.random.seed(int(time.clock()))

    # filter with probability:
    mask = np.random.uniform(0, 1, x.shape)
    mask[mask >= fraction] = 1.0
    mask[mask < fraction] = 0.0
    mask[x == 0.] = 0.

    outliers = np.random.normal(0, scale=scale, size=x.shape)

    x += np.multiply(outliers, mask)
    x[x < 0.] = 0.

    return x


class EarlyStopping(object):

    def __init__(self, start_epoch, n_epochs, decay_threshold=0.95, epoch_threshold=10):

        super(EarlyStopping, self).__init__()

        self.epoch = start_epoch - 1
        self.max_epoch = n_epochs

        self.decay_threshold = decay_threshold
        self.epoch_threshold = epoch_threshold

        self.last_update_epoch = 0
        self.last_update_loss = np.infty

        self.stop_iteration = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.stop_iteration:
            if self.epoch < self.max_epoch:
                self.epoch += 1
                return self.epoch

            else:
                raise StopIteration()

        else:
            raise StopIteration()

    def update(self, loss, epoch):

        if loss < 0:
            self.stop_iteration = True

        if loss/self.last_update_loss < self.decay_threshold:
            self.last_update_loss = loss
            self.last_update_epoch = epoch

        elif self.last_update_epoch < epoch - self.epoch_threshold:
            self.stop_iteration = True


def roll(x, n):
    return torch.cat((x[:, :, -n:], x[:, :, :-n]), dim=2)


if __name__ == '__main__':

    loss = 10

    stopper = EarlyStopping(n_epochs=10, epoch_threshold=2)

    for epoch in stopper:
        print(epoch)
        if epoch < 2:
            loss -= 1
            stopper.update(loss, epoch)
        else:
            stopper.update(loss, epoch)

    print('iteration worked')



def masking(x, values, threshold=0., option='ueq'):

    if option == 'leq':

        if x.dim() == 2:
            valid = (values <= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values <= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == 'geq':

        if x.dim() == 2:
            valid = (values >= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values >= threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == 'eq':

        if x.dim() == 2:
            valid = (values == threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values == threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]

    if option == 'ueq':

        if x.dim() == 2:
            valid = (values != threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid]
        if x.dim() == 3:
            valid = (values != threshold)[0, :, 0]
            xvalid = valid.nonzero()[:, 0]
            xmasked = x[:, xvalid, :]


    return xmasked