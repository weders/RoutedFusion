import os
import numpy as np
import logging
import datetime
import time
from datasets import SUNCG, Rays, Replica, ModelNet, ShapeNet
import random
import torch
import tensorboardX
import shutil
import json
import argparse

from multiprocessing import Process

from torch.utils.data import DataLoader

from torch import onnx

from weight_learning.config_end_to_end import config

if config.SETTINGS.machine == 'leonhard':
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

from weight_learning.model import FusionNet
from weight_learning.loss import OutlierTSDFLoss, BalancedLoss, RayLoss, DiffLoss, TSDFLoss, MaskTSDFLoss, UncertaintyTSDFLoss
from weight_learning.utils import add_axial_noise, EarlyStopping, MedianPool2d
from weight_learning.summaries import value_mean_summary, value_variance_summary, loss_summary, gradient_summary
from weight_learning.evaluator import TrainEvaluator, ValEvaluator
from weight_learning.metrics import *

from routing.model import UNet
from routing.loss import GradientWeightedDepthLoss, TSDFDepthLoss

from depth_completion.utils.transform import ToTensor
from datasets import SCANNET
from weight_learning.griddb import VolumeDB
from weight_learning.extractor import Extractor
from weight_learning.integrator import Integrator
from weight_learning.evaluation import eval
from routing.utils import load_config

from utils.config import load_config_yaml, save_checkpoint, save_config, save_code

from copy import copy
from time import sleep
from scipy.ndimage.filters import median_filter

# import matplotlib.pyplot as plt
# from graphics.utils import extract_mesh_marching_cubes
# from graphics.visualization import plot_mesh


def init_directories(config, create=True):

    # define all directories
    if config.SETTINGS.machine == 'local':
        log_dir = 'weight_learning/logs/{}/{}/{}'.format(config.SETTINGS.task, config.DATA.dataset, config.TIMESTAMP)
        model_dir = os.path.join('weight_learning/models', config.SETTINGS.task, config.DATA.dataset, config.TIMESTAMP)
        model_dir = os.path.expanduser(model_dir)

        if config.DATA.dataset == 'suncg':
            root_dir = '~/Polybox/Master/03-semester/01-master-thesis/05-data/01-suncg/suncg_v1'
        elif config.DATA.dataset == 'scannet':
            root_dir = '~/Polybox/Master/03-semester/01-master-thesis/05-data/03-scannet/scannet'

    elif config.SETTINGS.machine == 'workstation':
        log_dir = 'weight_learning/logs/{}/{}/{}'.format(config.SETTINGS.task, config.DATA.dataset, config.TIMESTAMP)
        model_dir = os.path.join('weight_learning/models', config.SETTINGS.task, config.DATA.dataset, config.TIMESTAMP)
        model_dir = os.path.expanduser(model_dir)
        preprocessing_dir = os.path.join('weight_learning',
                                         'models/pretrained/depth/suncg/')
        preprocessing_dir = os.path.expanduser(preprocessing_dir)
        if config.DATA.dataset == 'suncg':
            root_dir = '/local/home/data/suncg_v1'
        elif config.DATA.dataset == 'scannet':
            root_dir = '/local/home/data/scannet'

    elif config.SETTINGS.machine == 'leonhard':
        log_dir = '/cluster/scratch/weders/logs/001-real-time-depth-map-fusion/weight_learning/{}'.format(config.TIMESTAMP)
        model_dir = '/cluster/scratch/weders/models/001-real-time-depth-map-fusion/weight_learning/{}'.format(config.TIMESTAMP)

        if config.DATA.dataset == 'scannet':
            root_dir = '/cluster/scratch/weders/data/scannet'
        elif config.DATA.dataset == 'suncg':
            root_dir = '/cluster/scratch/weders/data/suncg/suncg_v1'
        elif config.DATA.dataset == 'replica':
            root_dir = '/cluster/scratch/weders/data/replica/apartment_0/habitat'
        elif config.DATA.dataset == 'modelnet':
            # root_dir = '/media/weders/HV620S/data/modelnet/ModelNet10'
            root_dir = '/cluster/project/infk/cvg/weders/data/modelnet/processed100'
        elif config.DATA.dataset == 'shapenet':
            # root_dir = '/media/weders/HV620S/data/modelnet/ModelNet10'
            root_dir = '/cluster/project/infk/cvg/weders/data/shapenet/processed'

    elif config.SETTINGS.machine == 'office':
        log_dir = '/local/home/projects/001-real-time-depth-map-fusion/logs/' \
                  'weight_learning/{}'.format(config.TIMESTAMP)
        model_dir = os.path.join('/local/home/projects/001-real-time-depth-map-fusion/models/' \
                                 'weight_learning/{}'.format(config.TIMESTAMP))

        if config.DATA.dataset == 'replica':
            root_dir = '/media/weders/TB2HD/data/replica/apartment_0/habitat'

        elif config.DATA.dataset == 'suncg':
            root_dir = '/local/home/data/001-real-time-depth-map-fusion/suncg'

        elif config.DATA.dataset == 'modelnet':
            # root_dir = '/media/weders/HV620S/data/modelnet/ModelNet10'
            root_dir = '/media/weders/HV620S/data/modelnet/processed100'

        elif config.DATA.dataset == 'shapenet':
            # root_dir = '/media/weders/HV620S/data/modelnet/ModelNet10'
            root_dir = '/media/weders/HV620S/data/shape-net/processed'

    # create directories
    if create:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    config.SETTINGS.root_dir = root_dir
    config.SETTINGS.model_dir = model_dir
    config.SETTINGS.log_dir = log_dir

    return config

def init_resume(config):

    if config.SETTINGS.machine == 'office':

        if config.RESUME.machine == 'cluster':

            config.RESUME.model_dir = '/local/home/cluster/models/001-real-time-depth-map-fusion/weight_learning'
            config.RESUME.model_dir = os.path.join(config.RESUME.model_dir, config.RESUME.model)
            load_config(config)
            config.RESUME.model_dir = config.RESUME.model_dir
            config.RESUME.log_dir = config.SETTINGS.log_dir

    init_directories(config)

def init_logger(config):

    logging.basicConfig(filename=os.path.join(config.SETTINGS.log_dir, 'training.logs'),
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    return logger


def mask_loss(est, gt, depth, indices=None):
    valid = torch.ones_like(depth)
    invalid = torch.zeros_like(depth)
    idxs = torch.arange(0, depth.shape[1])

    if torch.cuda.is_available():
        idxs = idxs.cuda()

    valid_rays = torch.where(depth != 0.,
                             valid,
                             invalid).byte().squeeze_(2)
    valid_rays = torch.masked_select(idxs, valid_rays)

    est_masked = torch.index_select(est, 1, valid_rays)
    gt_masked = torch.index_select(gt, 1, valid_rays)

    del idxs
    if indices is not None:
        idxs_masked = torch.index_select(indices, 1, valid_rays)
        return est_masked, gt_masked, idxs_masked

    return est_masked, gt_masked


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



def prediction(input, model, config):
    raise NotImplementedError

def sample_frames(dataset, n):

    p_chose = n / len(dataset)

    frames = []

    for frame in dataset:

        random.seed(time.clock())
        p = random.uniform(0, 1)

        if p > (1 - p_chose):
            frames.append(frame['frame_id'])

    return frames




def save_pipeline(config):

    save_code('weight_learning/loss.py', config.SETTINGS.log_dir)
    save_code('weight_learning/extractor.py', config.SETTINGS.log_dir)
    save_code('weight_learning/griddb.py', config.SETTINGS.log_dir)
    save_code('weight_learning/train.py', config.SETTINGS.log_dir)
    save_code('weight_learning/config_end_to_end.py', config.SETTINGS.log_dir)
    save_code('weight_learning/model.py', config.SETTINGS.log_dir)

def arg_parser():

    parser = argparse.ArgumentParser(description="Training of 3D fusion network")
    parser.add_argument('--config', required=True)

    args = parser.parse_args()
    return vars(args)

def train(config):

    config.TIMESTAMP = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    if config.RESUME.do:

        init_resume(config)
        config = init_directories(config)
        logger = init_logger(config)
        logger.info('{}'.format(config))

    else:
        # initialize all directories
        config = init_directories(config)
        logger = init_logger(config)
        logger.info('{}'.format(config))

    save_config(config)

    # create and initialize data
    if config.DATA.dataset == 'suncg':

        train_data = SUNCG(scene_list='weight_learning/list/suncg/current/{}'.format(config.DATA.set),
                           root_dir=config.SETTINGS.root_dir,
                           keys=config.DATA.modalities,
                           transform=ToTensor(),
                           n_samples=config.TRAINING.n_samples,
                           noise_scale=config.DATA.sigma_fraction)

        val_data = SUNCG(scene_list='weight_learning/list/suncg/current/val.v1.txt',
                         keys=config.DATA.modalities,
                         root_dir=config.SETTINGS.root_dir,
                         transform=ToTensor(),
                         n_samples=config.VALIDATION.n_samples,
                         noise_scale=config.DATA.sigma_fraction)

        train_tsdfdb = VolumeDB(scene_list='weight_learning/list/suncg/current/{}'.format(config.DATA.set),
                                dataset=train_data,
                                transform=ToTensor(),
                                initial_value=config.DATA.init_value)

        val_tsdfdb = VolumeDB(scene_list='weight_learning/list/suncg/current/val.v1.txt',
                              dataset=val_data,
                              transform=ToTensor(),
                              initial_value=config.DATA.init_value)

        rays = Rays()

        train_tsdfdb.to_tsdf(mode=config.MODEL.tsdf_mode)
        val_tsdfdb.to_tsdf(mode=config.MODEL.tsdf_mode)

        #train_tsdfdb.save('debug.npz')

    elif config.DATA.dataset == 'scannet':

        train_data = SCANNET(root_dir=config.SETTINGS.root_dir,
                             keys=config.DATA.modalities,
                             scene_list='weight_learning/list/scannet/debug.txt',
                             transform=ToTensor(),
                             n_samples=config.TRAINING.n_samples)

        val_data = SCANNET(root_dir=config.SETTINGS.root_dir,
                           keys=config.DATA.modalities,
                           scene_list='weight_learning/list/scannet/debug.txt',
                           transform=ToTensor(),
                           n_samples=config.TRAINING.n_samples)

        train_tsdfdb = VolumeDB(scene_list='weight_learning/list/scannet/debug.txt',
                                dataset=train_data,
                                transform=ToTensor())

    elif config.DATA.dataset == 'modelnet':

        train_data = ModelNet(config.SETTINGS.root_dir,
                              scene_list=config.DATA.train_scene_list,
                              transform=ToTensor(),
                              noise_scale=config.DATA.noise_scale,
                              outlier_scale=config.DATA.outlier_scale,
                              outlier_fraction=config.DATA.outlier_fraction,
                              grid_resolution=config.DATA.grid_resolution)

        val_data = ModelNet(config.SETTINGS.root_dir,
                            scene_list=config.DATA.val_scene_list,
                            transform=ToTensor(),
                            noise_scale=config.DATA.noise_scale,
                            outlier_scale=config.DATA.outlier_scale,
                            outlier_fraction=config.DATA.outlier_fraction,
                            grid_resolution=config.DATA.grid_resolution)

        train_tsdfdb = VolumeDB(scene_list=config.DATA.train_scene_list,
                                dataset=train_data,
                                transform=ToTensor(),
                                initial_value=config.DATA.init_value)
        val_tsdfdb = VolumeDB(scene_list=config.DATA.val_scene_list,
                                dataset=val_data,
                                transform=ToTensor(),
                                initial_value=config.DATA.init_value)

        train_tsdfdb.to_tsdf(mode=config.MODEL.tsdf_mode)
        val_tsdfdb.to_tsdf(mode=config.MODEL.tsdf_mode)

    elif config.DATA.dataset == 'shapenet':

        train_data = ShapeNet(config.SETTINGS.root_dir,
                              scene_list=config.DATA.train_scene_list,
                              transform=ToTensor(),
                              noise_scale=config.DATA.noise_scale,
                              outlier_scale=config.DATA.outlier_scale,
                              outlier_fraction=config.DATA.outlier_fraction,
                              grid_resolution=config.DATA.grid_resolution)

        val_data = ShapeNet(config.SETTINGS.root_dir,
                            scene_list=config.DATA.val_scene_list,
                            transform=ToTensor(),
                            noise_scale=config.DATA.noise_scale,
                            outlier_scale=config.DATA.outlier_scale,
                            outlier_fraction=config.DATA.outlier_fraction,
                            grid_resolution=config.DATA.grid_resolution)

        train_tsdfdb = VolumeDB(scene_list=config.DATA.train_scene_list,
                                dataset=train_data,
                                transform=ToTensor(),
                                initial_value=config.DATA.init_value)
        val_tsdfdb = VolumeDB(scene_list=config.DATA.val_scene_list,
                              dataset=val_data,
                              transform=ToTensor(),
                              initial_value=config.DATA.init_value)

        train_tsdfdb.to_tsdf(mode=config.MODEL.tsdf_mode)
        val_tsdfdb.to_tsdf(mode=config.MODEL.tsdf_mode)

    elif config.DATA.dataset == 'replica':

        train_data = Replica(root_dir=config.SETTINGS.root_dir,
                             frame_list='train_200.txt',
                             transform=ToTensor())

        val_data = Replica(root_dir=config.SETTINGS.root_dir,
                           frame_list='val_200.txt',
                           transform=ToTensor())

        train_tsdfdb = VolumeDB(scene_list=None,
                                dataset=train_data,
                                transform=ToTensor(),
                                initial_value=config.DATA.init_value)

        val_tsdfdb = VolumeDB(scene_list=None,
                              dataset=val_data,
                              transform=ToTensor(),
                              initial_value=config.DATA.init_value)

        rays = Rays()

        train_tsdfdb.to_tsdf(mode=config.MODEL.tsdf_mode)
        val_tsdfdb.to_tsdf(mode=config.MODEL.tsdf_mode)

        # for key in val_tsdfdb.scenes_gt.keys():
        #     grid = val_tsdfdb.scenes_gt[key].volume
        #     mesh = extract_mesh_marching_cubes(grid, level=-1.e-08)
        #     plot_mesh(mesh)

    train_loader = DataLoader(train_data,
                              batch_size=config.TRAINING.batch_size,
                              shuffle=False)

    val_loader = DataLoader(val_data, batch_size=config.VALIDATION.batch_size, shuffle=False)

    numbers = np.arange(0, len(val_data)-1)

    if config.VALIDATION.n_vis > len(val_data):
        config.VALIDATION.n_vis = len(val_data)-1
    vis_numbers = np.random.choice(numbers, config.VALIDATION.n_vis, replace=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_pipeline(config)

    # initialize model
    # tsdf_model = FusionNet(uncertainty=config.DATA.confidence,
    #                        output_uncertainty=(config.LOSS.loss == 'uncertainty' \
    #                                            or config.LOSS.loss == 'outlier'))

    if config.MODEL.dynamic:
        tsdf_model = FusionNet4()
    else:
        # tsdf_model = FusionNet3((config.LOSS.loss == 'uncertainty') or (config.LOSS.loss == 'outlier'))
        tsdf_model = FusionNet(confidence=config.DATA.confidence, n_points=config.MODEL.n_points)

    extractor = Extractor(mode=config.MODEL.mode, n_points=config.MODEL.n_points)
    extractor.eval()

    integrator = Integrator()
    integrator = integrator.to(device)
    # get computing resources
    tsdf_model = tsdf_model.to(device)
    extractor.to(device)

    # setup routing network
    if config.ROUTING.do:

        from routing.config import config as routing_config
        from routing.utils import load_config, load_checkpoint
        from routing.model import ConfidenceRouting

        routing_config.SETTINGS.machine = config.SETTINGS.machine
        routing_config.RESUME.model = config.ROUTING.model

        if config.SETTINGS.machine == 'leonhard':
            routing_config.SETTINGS.log_dir = '/cluster/scratch/weders/logs/001-real-time-depth-map-fusion/routing/{}'.format(
                config.TIMESTAMP)
            routing_config.SETTINGS.model_dir = '/cluster/scratch/weders/models/001-real-time-depth-map-fusion/routing/{}'.format(
                config.TIMESTAMP)
            routing_config.SETTINGS.root_dir = '/cluster/scratch/weders/data/suncg/suncg_v1'

        elif config.ROUTING.machine == 'cluster':
            routing_config.SETTINGS.log_dir = '/media/weders/HV620S/cluster/logs/001-real-time-depth-map-fusion/routing/{}'.format(
                config.TIMESTAMP)
            routing_config.SETTINGS.model_dir = '/media/weders/HV620S/cluster/models/001-real-time-depth-map-fusion/routing/{}'.format(
                config.TIMESTAMP)

        elif config.ROUTING.machine == 'office':
            routing_config.SETTINGS.log_dir = '/local/home/projects/001-real-time-depth-map-fusion/logs/routing/{}'.format(
                config.TIMESTAMP)
            routing_config.SETTINGS.model_dir = '/local/home/projects/001-real-time-depth-map-fusion/models/routing/{}'.format(
                config.TIMESTAMP)


        load_config(routing_config)

        routing_model = ConfidenceRouting(routing_config.MODEL.n_input_channels,
                                          routing_config.MODEL.contraction,
                                          Cout=1,
                                          depth=routing_config.MODEL.depth,
                                          batchnorms=routing_config.MODEL.normalization)

        if config.SETTINGS.machine == 'leonhard':
            path = '/cluster/scratch/weders/models/001-real-time-depth-map-fusion/routing'
            path = os.path.join(path, routing_config.RESUME.model)
            checkpoint = os.path.join(path, 'best.pth.tar')
            load_checkpoint(checkpoint, routing_model)

        if config.SETTINGS.machine == 'office':
            if config.ROUTING.machine == 'office':
                path = '/local/home/projects/001-real-time-depth-map-fusion/models/routing'
                path = os.path.join(path, routing_config.RESUME.model)
                checkpoint = os.path.join(path, 'best.pth.tar')
                load_checkpoint(checkpoint, routing_model)
            elif config.ROUTING.machine == 'cluster':
                path = '/media/weders/HV620S/cluster/models/001-real-time-depth-map-fusion/routing'
                path = os.path.join(path, routing_config.RESUME.model)
                checkpoint = os.path.join(path, 'best.pth.tar')
                load_checkpoint(checkpoint, routing_model)

        class Routing(torch.nn.Module):

            def __init__(self, model):

                super(Routing, self).__init__()

                self.model = model

            def forward(self, x):
                return self.model.forward(x)

        routing_model = routing_model.to(device)
    
    else:
        routing_model = None
        routing_config = None



    # detect GPU computing
    if torch.cuda.device_count() > 0:
        logger.info('using GPU computing')
        print("Let's use GPU computing")
    else:
        logger.info('no GPU available')
        print('No GPU available')

    # check whether multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        logger.info('using {} GPUs'.format(torch.cuda.device_count()))
        tsdf_model = torch.nn.DataParallel(tsdf_model)

    criterion_depth = TSDFDepthLoss()
    criterion_depth = criterion_depth.to(device)

    criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()

    if config.MODEL.mask:
        criterion3 = MaskTSDFLoss()
    elif config.LOSS.loss == 'uncertainty':
        criterion3 = UncertaintyTSDFLoss(l1=config.LOSS.l1,
                                         l2=config.LOSS.l2,
                                         cos=config.LOSS.cos)

    elif config.LOSS.loss == 'outlier':
       criterion3 = OutlierTSDFLoss()

    else:
        criterion3 = TSDFLoss(l1=config.LOSS.l1,
                              l2=config.LOSS.l2,
                              cos=config.LOSS.cos)

    if config.MODEL.dynamic:
        depth_criterion = GradientWeightedDepthLoss()
        depth_criterion = depth_criterion.to(device)

    criterion = criterion.to(device)
    criterion2 = criterion2.to(device)
    criterion3 = criterion3.to(device)

    l1_criterion = torch.nn.L1Loss(reduction='none')
    l2_criterion = torch.nn.MSELoss(reduction='none')
    cos_criterion = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='none')

    l1_criterion = l1_criterion.to(device)
    l2_criterion = l2_criterion.to(device)
    cos_criterion = cos_criterion.to(device)

    if config.ROUTING.do and config.ROUTING.train:
        from routing.loss import UncertaintyDepthLoss

        intermediate_criterion = UncertaintyDepthLoss()
        intermediate_criterion = intermediate_criterion.to(device)
        uncertainty_criterion = UncertaintyTSDFLoss()

    if config.ROUTING.do and config.ROUTING.train:
        optimizer = torch.optim.RMSprop(list(tsdf_model.parameters()) + list(routing_model.parameters()),
                                        config.OPTIMIZATION.lr,
                                        config.OPTIMIZATION.rho,
                                        config.OPTIMIZATION.eps,
                                        momentum=config.OPTIMIZATION.momentum,
                                        weight_decay=config.OPTIMIZATION.weight_decay)
    else:
        optimizer = torch.optim.RMSprop(tsdf_model.parameters(),
                                        config.OPTIMIZATION.lr,
                                        config.OPTIMIZATION.rho,
                                        config.OPTIMIZATION.eps,
                                        momentum=config.OPTIMIZATION.momentum,
                                        weight_decay=config.OPTIMIZATION.weight_decay)

        #confidence_parameters = []

        #for name, p in routing_model.named_parameters():

        #    if 'uncertainty' not in name:
        #        continue
        #    else:
        #        print(name)
        #        confidence_parameters.append(p)

        #optimizer_depth = torch.optim.RMSprop(confidence_parameters,
        #                                      1e-09,
        #                                      config.OPTIMIZATION.rho,
        #                                      config.OPTIMIZATION.eps,
        #                                      momentum=config.OPTIMIZATION.momentum,
        #                                      weight_decay=0.001)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=100,
                                                gamma=0.1)


    visualization_frames = sample_frames(train_data, 30)
    print(visualization_frames)
    # initialize tensorboard logger
    summary_writer = SummaryWriter(config.SETTINGS.log_dir)

    # initialize training iterator (early stopping)
    if config.SETTINGS.resume:
        training_iterator = EarlyStopping(config.SETTINGS.resume_epoch,
                                          config.TRAINING.n_epochs,
                                          config.TRAINING.decay_threshold,
                                          config.TRAINING.epoch_threshold)
    else:
        training_iterator = EarlyStopping(0,
                                          config.TRAINING.n_epochs,
                                          config.TRAINING.decay_threshold,
                                          config.TRAINING.epoch_threshold)

    n_batches = float(len(train_data)/config.TRAINING.batch_size)

    train_evaluator = TrainEvaluator(summary_writer, n_batches)
    val_evaluator = TrainEvaluator(summary_writer, len(val_data))
    # train_evaluator.add_metric('Output/nLargeValues', large_values)

    # load model and optimizer for resuming training
    if config.RESUME.do:
        path = config.RESUME.model_dir
        checkpoint = os.path.join(path, 'best.pth.tar')
        load_checkpoint(checkpoint, tsdf_model)

    elif config.PRETRAINED.do:

        if config.SETTINGS.machine == 'office':

            if config.PRETRAINED.machine == 'cluster':
                path = '/media/weders/HV620S/cluster/models/001-real-time-depth-map-fusion/weight_learning'
                path = os.path.join(path, config.PRETRAINED.model)
                checkpoint = os.path.join(path, 'best.pth.tar')

                load_checkpoint(checkpoint, tsdf_model)



    torch.autograd.set_detect_anomaly(True)

    # stopping criterias
    best_occ_acc = 0.
    best_iou = 0.
    best_l2 = np.infty
    best_l1 = np.infty
    best_mad = np.infty

    last_best = 0

    save_pipeline(config)

    if config.LOSS.loss == 'uncertainty' and config.PRETRAINED.do:

        for name, p in tsdf_model.named_parameters():

            if name.split('.')[0] != 'confidence_prediction':
                p.requires_grad = False
                print('switching off training for parameter', name)

    sigma = None
    mean = None

    for epoch in range(0, config.TRAINING.n_epochs):

        fused_data = None
        target_data = None

        if epoch == 200:
            for name, p in tsdf_model.named_parameters():
                p.requires_grad = True

        tsdf_model.train()

        # reset current TSDFs
        train_tsdfdb.reset()

        adversarial_frames = []
        adversarial_ids = []

        epoch_loss = {'MSE': 0.,
                      'L1': 0.,
                      'TSDF': 0.}

        pcl = []

        nmask = 0

        if config.LOSS.loss == 'uncertainty' and config.PRETRAINED.do and epoch > 20:

            for name, p in tsdf_model.named_parameters():

                if name.split('.')[0] != 'confidence_prediction':
                    p.requires_grad = True
                    print('switching on training for parameter', name)



        for i, batch in enumerate(train_loader):

            if batch['frame_id'][0] in visualization_frames:
                visualize = True
            else:
                visualize = False

            tsdf_model.eval()
            tsdf_model.train()

            batch_loss = {'MSE': 0.,
                          'L1': 0.,
                          'TSDF': 0.}

            batch['depth'][batch['original_mask'] == 0] = 0.

            output = pipeline(batch, extractor, routing_model, tsdf_model, integrator, train_tsdfdb,
                              device, config, routing_config,
                              mean=mean, sigma=sigma, mode='train')

            tsdf_fused = output['tsdf_fused']
            tsdf_target = output['tsdf_target']

            # if target_data is None:
            #     target_data = tsdf_target
            # else:
            #     target_data = torch.cat([target_data, tsdf_target], dim=1)
            #
            # if fused_data is None:
            #     fused_data = tsdf_fused
            # else:
            #     fused_data = torch.cat([fused_data, tsdf_fused], dim=1)

            tsdf_fused_unmasked = output['tsdf_fused_unmasked']
            tsdf_target_unmasked = output['tsdf_target_unmasked']

            #
            # routing_conf = output['confidence']

            if config.LOSS.loss == 'uncertainty':
                tsdf_conf_unmasked = output['tsdf_conf_unmasked']
                tsdf_unc = output['tsdf_unc']
                tsdf_conf = output['tsdf_conf']

                loss3 = criterion3.forward(tsdf_fused_unmasked, tsdf_target_unmasked, tsdf_conf_unmasked)

            elif config.LOSS.loss == 'outlier':

                tsdf_outliers = output['tsdf_outliers']
                # tsdf_outliers = masking(tsdf_outliers, output['filtered_frame'].view(1, 240*320, 1))

                # outlier_filter = torch.ones_like(tsdf_outliers) - tsdf_outliers
                #
                # tsdf_fused = outlier_filter * tsdf_fused
                # tsdf_target = outlier_filter * tsdf_target

                tsdf_outliers = tsdf_outliers.squeeze_(-1)
                loss3 = criterion3.forward(tsdf_fused_unmasked, tsdf_target_unmasked, tsdf_outliers, config.DATA.grid_resolution)
            else:

                loss3 = criterion3.forward(tsdf_fused, tsdf_target)

                if config.MODEL.dynamic:

                    depth_est = output['tsdf_depth'].unsqueeze_(0)
                    depth_target = batch[config.DATA.target].to(device).unsqueeze_(0)

                    if visualize:
                        summary_writer.add_image('Fusion/image{}'.format(batch['frame_id']),
                                                 depth_est[0])

                    loss3_depth = depth_criterion.forward(depth_est, depth_target)
                    loss3 = loss3 + loss3_depth

                # loss3 = criterion3.forward(tsdf_fused_unmasked, tsdf_target_unmasked)

            loss = criterion.forward(tsdf_fused, tsdf_target)
            loss2 = criterion2.forward(tsdf_fused, tsdf_target)

            if config.LOSS.loss == 'uncertainty':

                routing_conf = tsdf_conf

                print('confidence', torch.unique(routing_conf))

                tsdf_fused_filtered_high = masking(tsdf_fused, tsdf_conf, threshold=0.8, option='geq')
                tsdf_target_filtered_high = masking(tsdf_target, tsdf_conf, threshold=0.8, option='geq')
                tsdf_conf_filtered_high = masking(tsdf_conf, tsdf_conf, threshold=0.8, option='geq')

                tsdf_fused_filtered_low = masking(tsdf_fused, tsdf_conf, threshold=0.8, option='leq')
                tsdf_target_filtered_low = masking(tsdf_target, tsdf_conf, threshold=0.8, option='leq')
                tsdf_conf_filtered_low = masking(tsdf_conf, tsdf_conf, threshold=0.8, option='leq')

                n_low = torch.sum(torch.where(routing_conf < 0.8, torch.ones_like(routing_conf), torch.zeros_like(routing_conf)))
                n_high = 240*320 - n_low

                loss_1_high_conf = l1_criterion.forward(tsdf_fused_filtered_high, tsdf_target_filtered_high).sum()/n_high
                loss_2_high_conf = l2_criterion.forward(tsdf_fused_filtered_high, tsdf_target_filtered_high).sum()/n_high
                # loss_tsdf_high_conf = criterion3.forward(tsdf_fused_filtered_high, tsdf_target_filtered_high, tsdf_conf_filtered_high) * 240 * 320 / n_high

                loss_1_low_conf = l1_criterion.forward(tsdf_fused_filtered_low, tsdf_target_filtered_low).sum()/n_low
                loss_2_low_conf = l2_criterion.forward(tsdf_fused_filtered_low, tsdf_target_filtered_low).sum()/n_low
                # loss_tsdf_low_conf = criterion3.forward(tsdf_fused_filtered_low, tsdf_target_filtered_low,
                #                                          tsdf_conf_filtered_low)*240*320/n_low

                # print('L1', 'low:', loss_1_low_conf.item(), 'high:', loss_1_high_conf.item())
                # print('L2', 'low:', loss_2_low_conf.item(), 'high:', loss_2_high_conf.item())
                # print('TSDF', 'low:', loss_tsdf_low_conf.item(), 'high:', loss_tsdf_high_conf.item())

            tsdf_loss = output['loss'].unsqueeze_(1)
            tsdf_outlier = output['outlier'].unsqueeze_(1)
            #confidence = output['confidence'].unsqueeze_(1)
            # uncertainty = -torch.log(confidence)
            #depth = output['frame'].unsqueeze_(1)
            target = batch[config.DATA.target].to(device).unsqueeze_(1)

            if epoch > 1000:

                loss_depth = criterion_depth.forward(depth, target, uncertainty, tsdf_outlier, tsdf_loss)
                loss_depth.backward(retain_graph=True)
                optimizer_depth.step()
                optimizer_depth.zero_grad()

            if visualize:

                summary_writer.add_image('Train/TSDFLoss/{}'.format(batch['frame_id']), tsdf_loss.squeeze_(1), global_step=epoch)
                summary_writer.add_image('Train/TSDFOutlier/{}'.format(batch['frame_id']), tsdf_outlier.squeeze_(1), global_step=epoch)
                #summary_writer.add_image('Train/Confidence/{}'.format(batch['frame_id']), confidence.squeeze_(1), global_step=epoch)

            """Optimization"""
            if config.ROUTING.do:

                depth_target = batch[config.DATA.target].to(device)
                depth_target = depth_target.unsqueeze_(0)
                depth_est = output['depth'].squeeze_(-1).unsqueeze_(0)
                unc_est = output['uncertainty'].unsqueeze_(0).squeeze_(-1)

                mask = torch.where(torch.exp(-1.*unc_est) > 0.8, torch.ones_like(unc_est), torch.zeros_like(unc_est))

                # if visualize:
                #
                #     routing_conf = output['confidence'].view(1, 240*320).unsqueeze_(-1)
                #
                #     summary_writer.add_image('Routing/Estimate/{}'.format(batch['frame_id']),
                #                              depth_est[0], global_step=epoch)
                #     summary_writer.add_image('Routing/Confidence/{}'.format(batch['frame_id']),
                #                              torch.exp(-1.*unc_est)[0], global_step=epoch)
                #     summary_writer.add_image('Routing/Loss/{}'.format(batch['frame_id']),
                #                              torch.abs(depth_est - depth_target)[0], global_step=epoch)
                #     summary_writer.add_image('Routing/Mask/{}'.format(batch['frame_id']),
                #                              mask[0], global_step=epoch)
                #
                #     tsdf_fused_vis = torch.where(routing_conf > 0.8, tsdf_fused_unmasked, torch.zeros_like(tsdf_fused_unmasked))
                #     tsdf_target_vis = torch.where(routing_conf > 0.8, tsdf_target_unmasked, torch.zeros_like(tsdf_target_unmasked))
                #
                #     loss_1 = l1_criterion.forward(tsdf_fused_vis, tsdf_target_vis)
                #     loss_2 = l2_criterion.forward(tsdf_fused_vis, tsdf_target_vis)
                #
                #     # computing cosine distance
                #     x1 = torch.sign(tsdf_fused_vis)
                #     x2 = torch.sign(tsdf_target_vis)
                #     x1 = x1[:, :, :]
                #     x2 = x2[:, :, :]
                #
                #     label = torch.ones_like(x1)
                #     loss_cos = cos_criterion.forward(x1, x2, label)
                #
                #     loss_1 = loss_1.sum(dim=-1) / config.MODEL.n_points
                #     loss_2 = loss_2.sum(dim=-1) / config.MODEL.n_points
                #     loss_cos = loss_cos.sum(dim=-1) / config.MODEL.n_points
                #
                #     summary_writer.add_histogram('Fusion/L1/{}_hist'.format(batch['frame_id']),
                #                                  loss_1, global_step=epoch)
                #     summary_writer.add_histogram('Fusion/L2/{}_hist'.format(batch['frame_id']),
                #                                  loss_2, global_step=epoch)
                #     summary_writer.add_histogram('Fusion/Cos/{}_hist'.format(batch['frame_id']),
                #                                  loss_cos, global_step=epoch)
                #
                #     loss_1 /= torch.max(loss_1)
                #     loss_2 /= torch.max(loss_2)
                #     loss_cos /= torch.max(loss_cos)
                #
                #     loss_1 = loss_1.view(depth_est.shape)
                #     loss_2 = loss_2.view(depth_est.shape)
                #     loss_cos = loss_cos.view(depth_est.shape)
                #
                #     summary_writer.add_image('Fusion/L1/{}'.format(batch['frame_id']),
                #                              loss_1[0], global_step=epoch)
                #
                #     summary_writer.add_image('Fusion/L2/{}'.format(batch['frame_id']),
                #                              loss_2[0], global_step=epoch)
                #
                #     summary_writer.add_image('Fusion/Cos/{}'.format(batch['frame_id']),
                #                              loss_cos[0], global_step=epoch)
                #
                #     del tsdf_fused_vis, tsdf_target_vis, loss_1, loss_2, loss_cos, x1, x2
                #
                #     if config.LOSS.loss == 'uncertainty':
                #
                #         fusion_conf = output['tsdf_conf_unmasked'].view(depth_est.shape)
                #         summary_writer.add_image('Fusion/Confidence/{}'.format(batch['frame_id']),
                #                                  fusion_conf[0], global_step=epoch)
                #
                #         fusion_mask = torch.where(fusion_conf > 0.8, torch.ones_like(fusion_conf), torch.zeros_like(fusion_conf))
                #         summary_writer.add_image('Fusion/Mask/{}'.format(batch['frame_id']),
                #                                  fusion_mask[0], global_step=epoch)
                #
                #         del fusion_conf, fusion_mask, loss_1, loss_2, loss_cos

                del depth_target, depth_est, unc_est, mask

            if config.ROUTING.do and config.ROUTING.train:

                depth_target = batch[config.DATA.target].to(device)
                depth_target = depth_target.unsqueeze_(0)
                depth_est = output['depth'].squeeze_(-1).unsqueeze_(0)
                unc_est = output['uncertainty'].unsqueeze_(0)

                mask = torch.where(torch.exp(-1.*unc_est) > 0.8, torch.ones_like(unc_est), torch.zeros_like(unc_est))

                print(torch.unique(torch.exp(-1.*unc_est)))

                if visualize:
                    summary_writer.add_image('Routing/Estimate/{}'.format(batch['frame_id']),
                                             depth_est[0], global_step=epoch)
                    summary_writer.add_image('Routing/Confidence/{}'.format(batch['frame_id']),
                                             torch.exp(-1.*unc_est)[0], global_step=epoch)
                    summary_writer.add_image('Routing/Loss/{}'.format(batch['frame_id']),
                                             torch.abs(depth_est - depth_target)[0], global_step=epoch)
                    summary_writer.add_image('Routing/Mask/{}'.format(batch['frame_id']),
                                             mask[0], global_step=epoch)



                print('mean loss', torch.max(torch.abs(depth_est - depth_target)[0].mean()))

                inter_loss = intermediate_criterion.forward(depth_est, unc_est, depth_target)
                loss3 = uncertainty_criterion.forward(output['tsdf_fused_unmasked'],
                                                      output['tsdf_target_unmasked'],
                                                      output['confidence'].view(1, output['tsdf_target_unmasked'].shape[1], 1))
                l = loss3 + inter_loss
                l.backward()
            else:
                # accumulate gradients
                loss3.backward()

            batch_loss['MSE'] += loss.item()
            batch_loss['L1'] += loss2.item()
            batch_loss['TSDF'] += loss3.item()



            print('[EPOCH]:', epoch, '[TRAINING]', '[FRAME]', i, '/', len(train_data),
                  '[L1 LOSS]', batch_loss['L1'],
                  '[MSE LOSS]', batch_loss['MSE'],
                  '[TSDF LOSS]:', batch_loss['TSDF'])

            if config.TRAINING.clipping and epoch < 1000:
                torch.nn.utils.clip_grad_norm_(tsdf_model.parameters(),
                                               max_norm=1.,
                                               norm_type=2)

                if config.ROUTING.train and config.ROUTING.do:
                    torch.nn.utils.clip_grad_norm_(routing_model.parameters(),
                                                   max_norm=0.01,
                                                   norm_type=2)


            grads = {}

            for name, p in tsdf_model.named_parameters():
                if p.requires_grad:
                    grads[name] = p.grad.norm()
                    # print(name, p.grad.norm())

            if config.ROUTING.do and config.ROUTING.train:
                for name, p in routing_model.named_parameters():
                    if p.requires_grad:
                        grads[name] = p.grad.norm()
                        # print(name, p.grad.norm())

            if config.OPTIMIZATION.accumulate:
                if (i + 1) % config.OPTIMIZATION.n_acc_steps == 0 or i == n_batches - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    # update learning rate
                    scheduler.step()

            else:
                optimizer.step()
                optimizer.zero_grad()
                # update learning rate
                scheduler.step()


            del output, batch, loss, loss2, loss3, grads
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated(device) / 1e6)
            # for i in range(0, target_data.shape[2]):
        #     summary_writer.add_histogram('Fusion/Target/{}'.format(i),
        #                                  target_data[:, :, i], global_step=epoch)
        #     summary_writer.add_histogram('Fusion/Estimate/{}'.format(i),
        #                                  fused_data[:, :, i], global_step=epoch)
        #
        #
        # summary_writer.add_scalars('Gradients', grads, global_step=epoch)

        # if config.MODEL.mask:
        #     summary_writer.add_scalar('nMasks', nmask/n_batches, global_step=epoch)

        for key in train_tsdfdb.scenes_est.keys():

            grid = train_tsdfdb.scenes_est[key].volume

            eval(torch.Tensor(grid),
                 torch.clamp(torch.Tensor(train_tsdfdb.scenes_gt[key].volume), -0.1, 0.1),
                 torch.Tensor(train_tsdfdb.fusion_weights[key]),
                 step=epoch,
                 writer=summary_writer,
                 group='Train/{}'.format(key),
                 logger=logger)

        if(epoch % 5 == 0):
            for key in train_tsdfdb.scenes_est.keys():
                train_tsdfdb.save(config.SETTINGS.log_dir,
                                  key,
                                  epoch=epoch)

        # train_evaluator.write(epoch)

        for key in epoch_loss.keys():
            epoch_loss[key] /= n_batches

        loss_summary(summary_writer, epoch_loss, epoch)

        for i, id in enumerate(adversarial_ids):
            summary_writer.add_text('Data/AdversarialIDs/{}'.format(i), id[0], global_step=epoch)

        # summary_writer.add_scalar('Data/nAdversarialImages',
        #                           len(adversarial_frames),
        #                           global_step=epoch)

        # TODO: does not work anymore
        # for i in range(min(5, len(adversarial_frames))):
        #     summary_writer.add_image('Data/AdversarialImages/{}'.format(i),
        #                              adversarial_frames[i],
        #                              global_step=epoch)

        if config.EVAL.do and epoch % config.EVAL.epoch == 0:

            tsdf_model.eval()
            val_tsdfdb.reset()

            # current evaluation metrics
            current_iou = 0.
            current_acc = 0.
            current_l1 = 0.
            current_l2 = 0.
            current_mad = 0.

            for j, batch in enumerate(val_loader):
                print(batch['frame_id'])

                batch_loss = {'MSE': 0.,
                              'L1': 0.,
                              'TSDF': 0.}

                output = pipeline(batch, extractor, routing_model, tsdf_model, integrator, val_tsdfdb,
                                  device, config, routing_config, sigma=sigma, mean=mean, mode='val')

                tsdf_fused = output['tsdf_fused']
                tsdf_target = output['tsdf_target']


                """Computing Losses"""

                if config.LOSS.loss == 'uncertainty':
                    tsdf_unc = output['tsdf_unc']
                    tsdf_conf = output['tsdf_conf']
                    loss3 = criterion3.forward(tsdf_fused, tsdf_target, tsdf_conf)
                elif config.LOSS.loss == 'outlier':
                    tsdf_fused_unmasked = output['tsdf_fused_unmasked']
                    tsdf_target_unmasked = output['tsdf_target_unmasked']
                    tsdf_outliers = output['tsdf_outliers']

                    # tsdf_outliers = masking(tsdf_outliers, output['filtered_frame'].view(1, 240 * 320, 1))

                    # outlier_filter = torch.ones_like(tsdf_outliers) - tsdf_outliers
                    #
                    # tsdf_fused = outlier_filter * tsdf_fused
                    # tsdf_target = outlier_filter * tsdf_target

                    tsdf_outliers = tsdf_outliers.squeeze_(-1)


                    loss3 = criterion3.forward(tsdf_fused_unmasked, tsdf_target_unmasked, tsdf_outliers, config.DATA.grid_resolution)
                    del tsdf_outliers, tsdf_fused_unmasked, tsdf_target_unmasked
                else:
                    loss3 = criterion3.forward(tsdf_fused, tsdf_target)

                loss = criterion.forward(tsdf_fused, tsdf_target)
                loss2 = criterion2.forward(tsdf_fused, tsdf_target)

                batch_loss['MSE'] += loss.item()
                batch_loss['L1'] += loss2.item()
                batch_loss['TSDF'] += loss3.item()

                print('[EPOCH]:', epoch, '[VALIDATION]', '[FRAME]', j, '/', len(val_data),
                      '[L1 LOSS]', batch_loss['L1'],
                      '[MSE LOSS]', batch_loss['MSE'],
                      '[TSDF LOSS]:', batch_loss['TSDF'])

                del output, batch, batch_loss, loss, loss2, loss3, tsdf_fused, tsdf_target
                torch.cuda.empty_cache()
                print(torch.cuda.memory_allocated(device) / 1e6)

            for key in val_tsdfdb.scenes_est.keys():
                grid = val_tsdfdb.scenes_est[key].volume
                results = eval(torch.Tensor(grid),
                               torch.clamp(torch.Tensor(val_tsdfdb.scenes_gt[key].volume), -0.1, 0.1),
                               torch.Tensor(val_tsdfdb.fusion_weights[key]),
                               step=epoch,
                               writer=summary_writer,
                               group='Eval',
                               logger=logger)

                current_acc += results['occu_acc']
                current_iou += results['iou']
                current_l1 += results['l1']
                current_l2 += results['l2']
                current_mad += results['mad']

            for key in val_tsdfdb.scenes_est.keys():
                val_tsdfdb.save(config.SETTINGS.log_dir,
                                key,
                                epoch=epoch)

            current_acc /= len(val_tsdfdb.scenes_est.keys())
            current_iou /= len(val_tsdfdb.scenes_est.keys())
            current_l1 /= len(val_tsdfdb.scenes_est.keys())
            current_l2 /= len(val_tsdfdb.scenes_est.keys())
            current_mad /= len(val_tsdfdb.scenes_est.keys())

            if current_mad < best_mad and current_iou > 0.7:
                best_mad = copy(current_mad)
                is_best = True
                last_best = epoch
            else:
                is_best = False

            if current_acc > best_occ_acc:
                best_occ_acc = copy(current_acc)
                last_best = epoch
            if current_l1 < best_l1:
                best_l1 = copy(current_l1)
                last_best = epoch
            if current_l2 < best_l2:
                best_l2 = copy(current_l2)
                last_best = epoch

            save_checkpoint({'epoch': epoch,
                             'state_dict': tsdf_model.state_dict(),
                             'optim_dict': optimizer.state_dict()},
                            is_best=is_best,
                            checkpoint=config.SETTINGS.model_dir)

            if (epoch - last_best) > 30:
                print("Stopping training at", epoch, "due to no further improvement")
                break


if __name__ == '__main__':
    args = arg_parser()
    config = load_config_yaml(args['config'])
    print('Training with:', config)
    train(config)
