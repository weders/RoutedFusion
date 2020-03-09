import torch
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Script for testing RoutedFusion')

    parser.add_argument('--config', required=False)
    parser.add_argument('--experiment', required=False)
    parser.add_argument('--dataset', required=True)

    args = parser.parse_args()
    return vars(args)


def test(args, config):
    raise NotImplementedError


if __name__ == '__main__':

    # parse commandline arguments
    args = arg_parse()

    # load config
    if args['config']:
        config = utils.load_config(args['config'])
    elif args['experiment']:
        config = utils.load_config_from_experiment(args['experiment'])
    elif:
        raise Argu
