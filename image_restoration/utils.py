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

def to_psnr(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [measure.compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]

    return ssim_list