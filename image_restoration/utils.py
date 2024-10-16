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

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                          range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [measure.compare_ssim(pred_image_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True) for ind
                 in range(len(pred_image_list))]

    return ssim_list


def validation(config, net, val_data_loader, device, exp_name, support_data=None, save_tag=False):
    psnr_list = []
    ssim_list = []
    if config.stage == 0:
        for task_id, task in enumerate(TASK_DATASETS_TRAIN):
            sub_psnr_list, sub_ssim_list = [], []
            one_val_dataloader = val_data_loader[0][task]
            if config.meta_train:
                X_S, Y_S = support_data[task]
                X_S, Y_S = X_S.to(device), Y_S.to(device)
            for batch_id, val_data in enumerate(tqdm(one_val_dataloader)):
                X, Y = val_data

                with torch.no_grad():
                    if config.meta_train:
                        Y_pred = net(X_S, Y_S, X)
                    else:
                        X = from_6d_to_4d(X.to(device))
                        Y = from_6d_to_4d(Y.to(device))
                        Y_pred = net(X)

                # --- Calculate the average PSNR --- #
                sub_psnr_list.extend(calc_psnr(Y_pred, Y))

                # --- Calculate the average SSIM --- #
                sub_ssim_list.extend(calc_ssim(Y_pred, Y))

                # --- Save image --- #
                # if save_tag:
                #     save_image(Y_pred, exp_name)
            psnr_list.append(sum(sub_psnr_list) / len(sub_psnr_list))
            ssim_list.append(sum(sub_ssim_list) / len(sub_ssim_list))

    else:
        sub_psnr_list, sub_ssim_list = [], []
        if config.meta_train:
            X_S, Y_S = support_data
            X_S, Y_S = X_S.to(device), Y_S.to(device)
        for batch_id, val_data in enumerate(tqdm(val_data_loader)):
            X, Y = val_data
            X = X.to(device)
            Y = Y.to(device)

            with torch.no_grad():
                if config.meta_train:
                    Y_pred = net(X_S, Y_S, X)
                else:
                    X = from_6d_to_4d(X)
                    Y = from_6d_to_4d(Y)
                    Y_pred = net(X)

            # --- Calculate the average PSNR --- #
            sub_psnr_list.extend(calc_psnr(Y_pred, Y))

            # --- Calculate the average SSIM --- #
            sub_ssim_list.extend(calc_ssim(Y_pred, Y))
            break

            # --- Save image --- #
            # if save_tag:
            #     save_image(Y_pred, exp_name)
        psnr_list.append(sum(sub_psnr_list) / len(sub_psnr_list))
        ssim_list.append(sum(sub_ssim_list) / len(sub_ssim_list))

    return psnr_list, ssim_list


def validation_val(config, net, val_data_loader, device, savedir, support_data=None, save_tag=False):
    psnr_list = []
    ssim_list = []

    if config.meta_train:
        X_S, Y_S = support_data
        # print(X_S.shape, Y_S.shape)
        X_S, Y_S = X_S.to(device), Y_S.to(device)
    for batch_id, val_data in enumerate(tqdm(val_data_loader)):
        X, Y = val_data

        with torch.no_grad():
            if config.meta_train:
                Y_pred = net(X_S, Y_S, X)
            else:
                X = from_6d_to_4d(X.to(device))
                Y = from_6d_to_4d(Y.to(device))
                Y_pred = net(X)
        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(Y_pred, Y))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(Y_pred, Y))

        # --- Save image --- #
        if save_tag:
            # print()
            save_image(Y_pred, str(batch_id), savedir, batch_id)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim
