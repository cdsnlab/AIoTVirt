import math
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
import warnings

from utils import to_psnr, print_log, validation, load_model, tb_logging, model_save, validation_val, get_specific_param
from config import get_config
from models.reshape import *
from dataset.utils import TASK_DATASETS_TRAIN, TASK_DATASETS_TEST
from dataset.factory import get_train_dataloader, get_support_data, get_val_dataloaders, get_eval_dataloader, \
    get_finetune_dataloader, generate_support_data
from models.model_factory import get_model


def str2bool(v):
    if v == 'True' or v == 'true':
        return True
    elif v == 'False' or v == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


warnings.filterwarnings(action='ignore')
plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('--meta_train', default=True, type=str2bool)
parser.add_argument('--ddp', default=False, type=str2bool)
parser.add_argument('--stage', default=0, type=int, help='0:pretrain, 1:adaptation, 2:test')
parser.add_argument('--case', default=0, type=int, help='0-6')
parser.add_argument('--model_name', default='metaweather', type=str)
parser.add_argument('--exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('--checkpoint', help='path to checkpoint', type=str, default=None)

# --- DDP --- #
parser.add_argument('--local_rank', type=int)
parser.add_argument('--world_size', type=int)

args = parser.parse_args()

config = get_config(args, meta=args.meta_train)

# --- Distributed Data Parallel initialize --- #
if config.ddp:
    raise NotImplementedError('DDP is not supported yet.')

# --- Set Seed --- #
seed = 19
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\n'.format(config.lr, config.img_size,
                                                                                            config.batch_size,
                                                                                            config.val_batch_size))
# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')

# --- Define & Load the network --- #
net = get_model(config)

# if you face the error saying 'module.~' when u try to load, then uncomment this line.
net = nn.DataParallel(net, device_ids=device_ids)

config.specific_param = None
logdir = f'./experiments/{config.exp_name}/log'
savedir = f'./experiments/{config.exp_name}'
net = load_model(config, savedir, net)
if config.stage == 1:
    config.specific_param = 'bias_1shot'
    logdir = f'./experiments/{config.exp_name}/finetune/{config.specific_param}/{TASK_DATASETS_TEST[config.case]}/log'
    savedir = f'./experiments/{config.exp_name}/finetune/{config.specific_param}/{TASK_DATASETS_TEST[config.case]}'

# --- Build optimizer --- #
if config.optimizer == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), betas=(0.9, 0.999), lr=config.lr, weight_decay=config.weight_decay)
elif config.optimizer == 'adamw':
    # ! comment these lines !!!!!
    if config.specific_param is not None:
        config.specific_param = 'bias'
        parameters = get_specific_param(config, net)
        params = [{'params': parameters, 'lr': config.lr}]
        optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), lr=config.lr, weight_decay=config.weight_decay)
    # ! comment these lines !!!!!
    else:
        config.specific_param = 'except_enc'
        parameters = get_specific_param(config, net)
        params = [{'params': parameters, 'lr': config.lr}]
        optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), lr=config.lr, weight_decay=config.weight_decay)
        optimizer_enc = torch.optim.AdamW(net.module.encoder.parameters(), betas=(0.9, 0.999), lr=1e-5,
                                          weight_decay=config.weight_decay)

net = net.to(device)

# --- Logging --- #
writer = SummaryWriter(logdir)

# --- DataLoader --- #
if config.stage == 0:
    if config.meta_train:
        support_data = generate_support_data(config, data_path='support_data1.pth', split='train')
    else:
        support_data = None
    lbl_train_data_loader = get_train_dataloader(config)
    val_data_loader = get_val_dataloaders(config)
else:
    support_data = get_support_data(config, TASK_DATASETS_TEST[config.case], split='shots')
    support_data[0], support_data[1] = support_data[0].to(device), support_data[1].to(device)
    lbl_train_data_loader = get_finetune_dataloader(config, TASK_DATASETS_TEST[config.case], split='shots')
    # val_data_loader = get_val_dataloaders(config, support_data=support_data)
    test_loader = get_eval_dataloader(config, task=TASK_DATASETS_TEST[config.case], split='test', mode='resize')
