import torch
import argparse
import datetime

from tqdm import tqdm

from utils.setup import *
from utils.loading import load_config_from_yaml
from utils.loss import FusionLoss

from modules.model import FusionNet
from modules.routing import ConfidenceRouting
from modules.pipeline import Pipeline


def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config')
    parser.add_argument('--experiment')

    args = parser.parse_args()
    return vars(args)


def train_fusion(args):

    config = load_config_from_yaml(args['config'])

    config.TIMESTAMP = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    # get workspace
    workspace = get_workspace(config)

    # save config before training
    workspace.save_config(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.MODEL.device = device


    # get datasets
    # get train dataset
    train_data_config = get_data_config(config, mode='train')
    train_dataset = get_data(config.DATA.dataset, train_data_config)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               config.TRAINING.train_batch_size)

    # get val dataset
    val_data_config = get_data_config(config, mode='val')
    val_dataset = get_data(config.DATA.dataset, val_data_config)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             config.TRAINING.val_batch_size)

    # get database
    # get train database
    train_database = get_database(train_dataset, config, mode='train')
    val_database = get_database(val_dataset, config, mode='val')

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline = pipeline.to(device)

    # optimization
    criterion = FusionLoss(config)

    optimizer = torch.optim.RMSprop(pipeline._fusion_network.parameters(),
                                    config.OPTIMIZATION.lr,
                                    config.OPTIMIZATION.rho,
                                    config.OPTIMIZATION.eps,
                                    momentum=config.OPTIMIZATION.momentum,
                                    weight_decay=config.OPTIMIZATION.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=config.OPTIMIZATION.scheduler.step_size,
                                                gamma=config.OPTIMIZATION.scheduler.gamma)

    # define some parameters
    n_batches = float(len(train_dataset)/config.TRAINING.train_batch_size)

    # evaluation metrics
    best_iou = 0.

    for epoch in range(0, config.TRAINING.n_epochs):

        print('Training on epoch {}/{}'.format(epoch, config.TRAINING.n_epochs))

        pipeline.train()

        # resetting databases before each epoch starts
        train_database.reset()
        val_database.reset()

        for i, batch in tqdm(enumerate(train_loader), total=len(train_dataset)):

            # put all data on GPU
            batch = transform.to_device(batch, device)

            # put original mask
            batch['mask'] = batch['original_mask']

            # fusion pipeline
            output = pipeline.fuse_training(batch, train_database, device)

            tsdf_fused = output['tsdf_fused']
            tsdf_target = output['tsdf_target']

            # optimization
            loss = criterion(tsdf_fused, tsdf_target)
            loss.backward()

            if config.TRAINING.clipping:
                torch.nn.utils.clip_grad_norm_(pipeline._fusion_network.parameters(),
                                               max_norm=1.,
                                               norm_type=2)

            if (i + 1) % config.OPTIMIZATION.accumulation_steps == 0 or i == n_batches - 1:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # zero out all grads
        optimizer.zero_grad()

        train_database.filter()
        train_eval = train_database.evaluate()
        train_database.save_to_workspace(workspace)
        print(train_eval)

        pipeline.eval()

        # validation step - fusion
        for i, batch in tqdm(enumerate(val_loader), total=len(val_dataset)):

            # put all data on GPU
            batch = transform.to_device(batch, device)

            # put original mask
            batch['mask'] = batch['original_mask']

            # fusion pipeline
            pipeline.fuse(batch, val_database, device)

        val_database.filter()
        val_eval = val_database.evaluate()
        print(val_eval)

        # check if current checkpoint is best
        if val_eval['iou'] >= best_iou:
            is_best = True
            best_iou = val_eval['iou']
            workspace.log('found new best model with iou {} at epoch {}'.format(best_iou, epoch),
                          mode='val')
        else:
            is_best = False

        # save models
        train_database.save_to_workspace(workspace)
        val_database.save_to_workspace(workspace)

        # save checkpoint
        workspace.save_model_state({'pipeline_state_dict': pipeline.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'epoch': epoch},
                                   is_best=is_best)

if __name__ == '__main__':
    args = arg_parser()
    train_fusion(args)
