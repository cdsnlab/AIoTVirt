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

# set seed
seed = 19
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('crop_size: {}\nval_batch_size: {}\n'.format(config.img_size, config.eval_batch_size))

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = get_model(config)

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load & record --- #
logdir = f'./experiments/{config.exp_name}/finetune/bias_1shot/{TASK_DATASETS_TEST[config.case]}/log'
savedir = f'./experiments/{config.exp_name}/finetune/bias_1shot/{TASK_DATASETS_TEST[config.case]}'

try:
    if config.checkpoint is not None:
        net.load_state_dict(torch.load(config.checkpoint))
        print('--- weight loaded ---')
    else:
        net.load_state_dict(torch.load(f'{savedir}/latest_finetune'))
        print('--- weight loaded for testing---')
except:
    print('--- no weight loaded ---')
    sys.exit(0)

test_loader = get_eval_dataloader(config, task=TASK_DATASETS_TEST[config.case], split='test', mode='resize')
if config.meta_train:
    support_data = get_support_data(config, TASK_DATASETS_TEST[config.case], split='shots')
    support_data[0], support_data[1] = support_data[0].to(device), support_data[1].to(device)
else:
    support_data = None

# # --- Previous PSNR and SSIM in testing --- #
net.eval()

# -------------------------------------------------------------------------------------------------------------
eval_psnr, eval_ssim = validation_val(config, net, test_loader, device, savedir, support_data, True)

print(f'Eval_PSNR: {eval_psnr:.4f}, Eval_SSIM: {eval_ssim:.5f}')
# with open( os.path.join(logdir,'test_{}.txt'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))), 'a') as f:
#     print(f'Eval_PSNR: {eval_psnr:.4f}, Eval_SSIM: {eval_ssim:.5f}', file=f)
