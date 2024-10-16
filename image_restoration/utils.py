import time
import sys
import torch
import torch.nn.functional as F
import torchvision.utils as utils
import torch.distributed as dist
import torch.nn as nn
from math import log10
from skimage import measure
import cv2
import os

from tqdm import tqdm
from einops import repeat

import skimage
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import pdb
from models.reshape import *
from dataset.utils import TASK_DATASETS_TEST, TASK_DATASETS_TRAIN

# from models.Restormer.Restormer import LayerNorm
from models.NAFNet.arch_util import LayerNorm2d


def calc_psnr(im1, im2):
    if len(im1.shape) == 3:
        im1 = im1.unsqueeze(0)
    elif len(im1.shape) == 6:
        im1 = from_6d_to_4d(im1)
        im2 = from_6d_to_4d(im2)
    im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_psnr(im1_y, im2_y, data_range=1.)]
    return ans


def calc_ssim(im1, im2):
    if len(im1.shape) == 3:
        im1 = im1.unsqueeze(0)
    elif len(im1.shape) == 6:
        im1 = from_6d_to_4d(im1)
        im2 = from_6d_to_4d(im2)
    im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_ssim(im1_y, im2_y, data_range=1.)]
    return ans
