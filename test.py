import torch
import argparse
import os

import numpy as np

from utils import loading
from utils import setup
from utils import transform

from modules.extractor import Extractor
from modules.integrator import Integrator
from modules.model import FusionNet
from modules.routing import ConfidenceRouting
from modules.functions import pipeline

from tqdm import tqdm

# from graphics.utils import extract_mesh_marching_cubes
# from graphics.visualization import plot_mesh

def arg_parse():
    parser = argparse.ArgumentParser(description='Script for testing RoutedFusion')

    parser.add_argument('--config', required=False)
    parser.add_argument('--experiment', required=False)
    parser.add_argument('--dataset', required=False)
    parser.add_argument('--routing-model', required=False)
    parser.add_argument('--fusion-model', required=False)

    args = parser.parse_args()

    return vars(args)


def test(args, config):

    # define data configuration
    config.DATA.root_dir = '/media/weders/HV620S/data/shape-net/processed'
    config.DATA.resy = 240
    config.DATA.resx = 320
    config.DATA.scene_list = 'lists/shapenet/train.test3.txt'
    config.DATA.transform = transform.ToTensor()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config.MODEL.confidence = False
    config.MODEL.uncertainty = False
    config.MODEL.device = device

    # get test dataset
    dataset = setup.get_data(args['dataset'], config)
    loader = torch.utils.data.DataLoader(dataset)

    # get test database
    database = setup.get_database(dataset, config)
    database.to_tsdf()

    # setup pipeline
    extractor = Extractor(config.MODEL)
    integrator = Integrator(config.MODEL)
    fusion = FusionNet(config.MODEL)
    routing = ConfidenceRouting(1, 64, 1, 1, False)

    extractor = extractor.to(device)
    integrator = integrator.to(device)
    fusion = fusion.to(device)
    routing = routing.to(device)

    # load trained components of pipeline
    loading.load_model(args['fusion_model'], fusion)
    loading.load_model(args['routing_model'], routing)

    for i, batch in tqdm(enumerate(loader), total=len(dataset)):

        scene_id = batch['scene_id'][0]

        entry = database[scene_id]

        # put all data on GPU
        entry = transform.to_device(entry, device)
        batch = transform.to_device(batch, device)

        # put original mask
        batch['mask'] = batch['original_mask']

        # fusion pipeline
        tsdf_grid, weights_grid = pipeline(batch, entry,
                                           routing, extractor, fusion, integrator,
                                           config)

        # update database
        database.scenes_est[scene_id]._volume = tsdf_grid.cpu().detach().numpy()
        database.fusion_weights[scene_id] = weights_grid.cpu().detach().numpy()

    database.filter()

    acc_tot = 0.

    for key in database.scenes_est.keys():

        tsdf_grid_est = torch.Tensor(database.scenes_est[key]._volume)
        tsdf_grid_gt = torch.Tensor(database.scenes_gt[key]._volume)
        tsdf_weights = database.fusion_weights[key]

        tsdf_grid_est[np.where(tsdf_weights < 3)] = 0.
        tsdf_grid_gt[np.where(tsdf_weights < 3)] = 0.


        database.save(args['output_dir'], key, groundtruth=True)

        # compute accuracy
        mask_gt = torch.where(torch.abs(tsdf_grid_gt) < 0.05, torch.ones_like(tsdf_grid_gt),
                              torch.zeros_like(tsdf_grid_gt)).int()
        mask_est = torch.where(torch.abs(tsdf_grid_est) < 0.05, torch.ones_like(tsdf_grid_est),
                               torch.zeros_like(tsdf_grid_est)).int()

        mask = mask_gt | mask_est
        mask = mask.float()
        est = mask * tsdf_grid_est
        gt = mask * tsdf_grid_gt

        est_p = torch.where(est < 0, torch.ones_like(est),
                            torch.zeros_like(est)).int()
        gt_p = torch.where(gt < 0, torch.ones_like(gt),
                           torch.zeros_like(gt)).int()
        est_n = torch.where(est > 0, torch.ones_like(est),
                            torch.zeros_like(est)).int()
        gt_n = torch.where(gt > 0, torch.ones_like(gt),
                           torch.zeros_like(gt)).int()

        tp = torch.sum(est_p & gt_p).item()
        fp = torch.sum(est_p).item() - tp
        tn = torch.sum(est_n & gt_n).item()
        fn = torch.sum(est_n).item() - tn

        sum = tp + fp + fn + tn
        if sum == 0.:
            sum = 1.0
            print('invalid volume')

        accuracy = (tp + tn) / sum

        accuracy = 100. * accuracy
        print(accuracy)
        acc_tot += accuracy

    print('final acc', acc_tot / 58.)


        # mesh = extract_mesh_marching_cubes(tsdf_grid, level=-1.e-08)
        # plot_mesh(mesh)


if __name__ == '__main__':

    # parse commandline arguments
    args = arg_parse()

    args['dataset'] = "ShapeNet"
    args['experiment'] = '/media/weders/HV620S/cluster/models/001-real-time-depth-map-fusion/weight_learning/191113-200403'
    args['routing_model'] = '/media/weders/HV620S/cluster/models/001-real-time-depth-map-fusion/routing/191107-220328/best.pth.tar'
    args['fusion_model'] = '/media/weders/HV620S/cluster/models/001-real-time-depth-map-fusion/weight_learning/191113-200403/best.pth.tar'
    args['output_dir'] = 'output'

    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    # load config
    if args['config']:
        config = loading.load_config(args['config'])
    elif args['experiment']:
        config = loading.load_experiment(args['experiment'])
    elif args['routing_model']:
        pass
    elif args['fusion_model']:
        pass
    else:
        raise ValueError('Missing configuration: Please either specify config or experiment. ')

    test(args, config)
