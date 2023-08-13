import logging
import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import tensorboardX
import torch.optim as optim
import wandb
from copy import deepcopy


def init_workspace(args):
    args.log = os.path.join(args.result, 'log', args.doc)
    args.checkpoint = os.path.join(args.result, 'checkpoint', args.doc)
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)


def init_monitor(args, wandb_config=None):
    tb_logger = None
    if args.monitor in ['wandb', 'tensorboard']:
        if args.monitor == 'wandb':
            init_wandb(project_name=args.doc, config=wandb_config)
        elif args.monitor == 'tensorboard':
            tb_logger = tensorboard_logger(args=args)
    else:
        raise AssertionError('two visualization options can be selected: [wandb, tensorboard]')
    return tb_logger


# init wandb
def init_wandb(project_name='my_project', config=None):
    if config is None:
        wandb.init(project=project_name)
    else:
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            # track hyperparameters and run metadata
            config=config
            # sample
            # config={
            #     "learning_rate": 0.02,
            #     "architecture": "CNN",
            #     "dataset": "CIFAR-100",
            #     "epochs": 10,
            # }
        )


def tensorboard_logger(args):
    # tensorboard
    tb_path = os.path.join(args.result, 'tensorboard', args.doc)
    if os.path.exists(tb_path):
        shutil.rmtree(tb_path)
    os.makedirs(tb_path)
    return tensorboardX.SummaryWriter(log_dir=tb_path)


def set_logger(args):
    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    # formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)


def get_optimizer(config, model):
    new_config = deepcopy(config)
    # optim
    optim_params = vars(new_config.optim)
    if optim_params['optimizer'] == 'Adagrad':
        del optim_params['optimizer']
        optimizer = optim.Adagrad(model.parameters(), **optim_params)
    elif optim_params['optimizer'] == 'Adam':
        del optim_params['optimizer']
        optimizer = optim.Adam(model.parameters())
    else:
        raise AssertionError('According to the original paper, you should use "Adagrad" as the optimizer')

    return optimizer


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def prepare_device(keep_reproducibility=False):
    if keep_reproducibility:
        logging.info("Using CuDNN deterministic mode in the experiment.")
        torch.backends.cudnn.benchmark = False  # ensures that CUDA selects the same convolution algorithm each time
        # torch.set_deterministic(True)  # configures PyTorch only to use deterministic implementation
    else:
        torch.backends.cudnn.benchmark = True
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # return torch.device("cpu")


def dict_merge(dictionary: dict):
    new_dict = deepcopy(dictionary)
    merge_dict = {}
    try:
        for key, value in new_dict.items():
            if isinstance(value, dict):
                merge_dict = dict(**merge_dict, **value)
            else:
                merge_dict[key] = value
    except TypeError:
        print("Exists the conflict key in the config sub-dict. ")
    return merge_dict