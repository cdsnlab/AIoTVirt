import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import numpy as np
import random

from config import get_config
from utils import validation_val
from dataset.utils import TASK_DATASETS_TEST
from dataset.factory import get_support_data, get_eval_dataloader
from models.model_factory import get_model

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('--meta_train', default=True, type=str2bool)
parser.add_argument('--stage', default=2, type=int, help='0:pretrain, 1:adaptation, 2:test')
parser.add_argument('--case', default=0, type=int, help='0-6')
parser.add_argument('--model_name', default='metaweather', type=str)
parser.add_argument('--exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('--checkpoint', help='path to checkpoint', type=str, default=None)

args = parser.parse_args()

config = get_config(args, meta=args.meta_train)


#set seed
seed = 19
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('crop_size: {}\nval_batch_size: {}\n'.format(config.img_size, config.eval_batch_size))