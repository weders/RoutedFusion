import torch
import argparse

from utils import loading
from utils import setup
from utils import transform

from modules.extractor import Extractor
from modules.integrator import Integrator
from modules.model import FusionNet
from modules.functions import pipeline_clean

from tqdm import tqdm

from graphics.utils import extract_mesh_marching_cubes
from graphics.visualization import plot_mesh

def arg_parse():
    parser = argparse.ArgumentParser(description='Script for testing RoutedFusion')

    parser.add_argument('--config', required=False)
    parser.add_argument('--experiment', required=False)
    parser.add_argument('--dataset', required=True)

    args = parser.parse_args()

    return vars(args)


def test(args, config):

    # define data configuration
    config.DATA.root_dir = '/media/weders/HV620S/data/shape-net/processed'
    config.DATA.resy = 240
    config.DATA.resx = 320
    config.DATA.scene_list = 'lists/shapenet/train.test3.txt'
    config.DATA.transform = transform.ToTensor()

    device = torch.device("cpu")
    config.MODEL.confidence = False
    config.MODEL.uncertainty = False
    config.MODEL.device = device

    # get test dataset
    dataset = setup.get_data(args['dataset'], config)
    loader = torch.utils.data.DataLoader(dataset)

    # get test database
    database = setup.get_database(dataset, config)

    # setup pipeline
    extractor = Extractor(config.MODEL)
    integrator = Integrator()
    fusion = FusionNet(config.MODEL)

    import matplotlib.pyplot as plt


    for i, batch in tqdm(enumerate(loader), total=len(dataset)):

        scene_id = batch['scene_id'][0]

        entry = database[scene_id]

        # put original mask
        batch['mask'] = batch['original_mask']

        # fusion pipeline
        tsdf_grid, weights_grid = pipeline_clean(batch, entry,
                                                 None, extractor, fusion, integrator,
                                                 config)

        # update database
        database.scenes_est[scene_id]._volume = tsdf_grid.detach().numpy()
        database.fusion_weights[scene_id] = weights_grid.detach().numpy()

    database.filter()

    for key in database.scenes_est.keys():
        tsdf_grid = database.scenes_est[key]._volume
        mesh = extract_mesh_marching_cubes(tsdf_grid, level=-1.e-08)
        plot_mesh(mesh)


if __name__ == '__main__':

    # parse commandline arguments
    args = arg_parse()

    # load config
    if args['config']:
        config = loading.load_config(args['config'])
    elif args['experiment']:
        config = loading.load_experiment(args['experiment'])
    else:
        raise ValueError('Missing configuration: Please either specify config or experiment. ')

    test(args, config)