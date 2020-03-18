import torch

from dataset import ShapeNet
from modules.griddb import VolumeDB

def get_data(dataset, config):
    return eval(dataset)(config.DATA)

def get_database(dataset, config):
    return VolumeDB(dataset, config.DATA)