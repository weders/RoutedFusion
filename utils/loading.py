import yaml
import json
import os
import torch

from easydict import EasyDict


def load_config_from_yaml(path):
    """
    Method to load the config file for
    neural network training
    :param path: yaml-filepath with configs stored
    :return: easydict containing config
    """
    c = yaml.load(open(path))
    config = EasyDict(c)

    return config


def load_config_from_json(path):
    """
    Method to load the config file
    from json files.
    :param path: path to json file
    :return: easydict containing config
    """
    with open(path, 'r') as file:
        data = json.load(file)
    config = EasyDict(data)
    return config


def load_experiment(path):
    """
    Method to load experiment from path
    :param path: path to experiment folder
    :return: easydict containing config
    """
    path = os.path.join(path, 'config.json')
    config = load_config_from_json(path)
    return config


def load_config(path):
    """
    Wrapper method around different methods
    loading config file based on file ending.
    """

    if path[-4:] == 'yaml':
        return load_config_from_yaml(path)
    elif path[-4:] == 'json':
        return load_config_from_json(path)
    else:
        raise ValueError('Unsupported file format for config')


def load_model(file, model):

    checkpoint = file

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    except:
        print('loading model partly')
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}
        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())

def load_pipeline(file, model):

    checkpoint = file

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['pipeline_state_dict'])
    except:
        print('loading model partly')
        pretrained_dict = {k: v for k, v in checkpoint['pipeline_state_dict'].items() if k in model.state_dict()}
        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    except:
        print('loading model partly')
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}
        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
