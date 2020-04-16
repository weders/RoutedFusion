import torch

from dataset import ShapeNet

from modules.griddb import VolumeDB

from utils import transform

from easydict import EasyDict
from copy import copy


def get_data_config(config, mode):

    data_config = copy(config.DATA)

    if mode == 'train':
        data_config.scene_list = data_config.train_scene_list
    elif mode == 'val':
        data_config.scene_list = data_config.val_scene_list
    elif mode == 'test':
        data_config.scene_list = data_config.test_scene_list

    data_config.transform = transform.ToTensor()

    return data_config


def get_data(dataset, config):

    try:
        return eval(dataset)(config.DATA)
    except AttributeError:
        return eval(dataset)(config)


def get_database(dataset, config):
    return VolumeDB(dataset, config.DATA)
