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


def save_image(pred_image, image_name, savedir, batch_id):
    if os.path.isdir(os.path.join(savedir, 'outputs')) is False:
        os.makedirs(os.path.join(savedir, 'outputs'))
        # import pdb; pdb.set_trace()
    if len(pred_image.shape) == 6:
        pred_image = from_6d_to_4d(pred_image)

    utils.save_image(pred_image[0], '{}/outputs/{}.png'.format(savedir, batch_id))


def print_log(epoch, num_epochs, train_psnr, val_psnr, val_ssim, exp_name):
    print('Epoch [{0}/{1}], Train_PSNR:{2:.2f}, Val_PSNR:{3:.2f}, Val_SSIM:{4:.4f}'
          .format(epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(exp_name), 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Train_PSNR: {3:.2f}, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, lr_decay=0.5):
    # --- Decay learning rate --- #
    step = 100

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))


def load_model(config, savedir, net):
    if config.stage == 0:
        try:
            if config.checkpoint is not None:
                net.load_state_dict(torch.load(config.checkpoint))
                print('--- weight loaded for continuing pretrain ---')
            elif config.exp_name is not None:
                net.load_state_dict(torch.load('{}/best'.format(savedir)))
                print('--- weight loaded for continuing pretrain ---')
            else:
                print('--- no weight loaded ---')
        except:
            print('--- no weight loaded ---')
    elif config.stage == 1:
        try:
            # net.load_state_dict(torch.load('{}/best'.format(savedir)))
            load_model_finetune(config, savedir, net)
            print('--- weight loaded for finetuning---')
        except:
            print('--- no weight loaded ---')
            if config.model_name != 'metaweather':
                sys.exit(0)
    return net


def load_model_finetune(config, savedir, net):
    if config.checkpoint is not None:
        state_dict = torch.load(config.checkpoint)
    elif config.exp_name is not None:
        state_dict = torch.load('{}/best'.format(savedir))

    try:
        net.load_state_dict(state_dict, strict=True)
    except:
        bias_parameters = [f'model.{name}' for name in net.model.bias_parameter_names()]
        for key in state_dict.keys():
            if key in bias_parameters:
                state_dict[key] = torch.zeros_like(state_dict[key][0])

                if config.n_tasks is not None and config.n_tasks > 0:
                    state_dict[key] = repeat(state_dict[key], '... -> T ...', T=config.n_tasks)

        net.load_state_dict(state_dict)


def tb_logging(config, writer, psnr_list, ssim_list, eval_psnr, eval_ssim, global_step, lr_scheduler=None):
    val_psnr = sum(psnr_list) / len(psnr_list)
    val_ssim = sum(ssim_list) / len(ssim_list)
    if config.stage == 0:
        writer.add_scalar(TASK_DATASETS_TRAIN[0] + 'val_psnr', psnr_list[0], global_step=global_step)
        writer.add_scalar(TASK_DATASETS_TRAIN[1] + 'val_psnr', psnr_list[1], global_step=global_step)
        writer.add_scalar(TASK_DATASETS_TRAIN[2] + 'val_psnr', psnr_list[2], global_step=global_step)
        writer.add_scalar(TASK_DATASETS_TRAIN[3] + 'val_psnr', psnr_list[3], global_step=global_step)
        writer.add_scalar('val_psnr', val_psnr, global_step=global_step)
        writer.add_scalar('val_ssim', val_ssim, global_step=global_step)
        print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
    elif config.stage == 1:
        writer.add_scalar('adapt' + '_val_psnr', psnr_list[0], global_step=global_step)
        writer.add_scalar('adapt' + '_val_ssim', ssim_list[0], global_step=global_step)
        writer.add_scalar('adapt' + '_eval_psnr', eval_psnr, global_step=global_step)
        writer.add_scalar('adapt' + '_eval_ssim', eval_ssim, global_step=global_step)
        print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
        print('eval_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(eval_psnr, eval_ssim))
    if lr_scheduler is not None:
        writer.add_scalar('lr', lr_scheduler.get_lr()[0], global_step=global_step)


def model_save(config, net, savedir):
    last_name = 'best' if config.stage == 0 else 'best_finetune'

    if config.ddp is True:
        if dist.get_rank() == 0:
            torch.save(net.state_dict(), '{}/{}'.format(savedir, last_name))
            print('model saved')
    else:
        torch.save(net.state_dict(), '{}/{}'.format(savedir, last_name))
        print('model saved')


def get_specific_param(config, model):
    def isLN(m: nn.Module):
        return isinstance(m, LayerNorm2d) or isinstance(m, nn.LayerNorm)
        # return isinstance(m, LayerNorm2d) or isinstance(m, LayerNorm) or isinstance(m, nn.LayerNorm)

    # import pdb; pdb.set_trace()
    if config.specific_param == 'LN':
        params = []
        for nm, m in model.named_modules():
            if isLN(m):
                for np, p in m.named_parameters():
                    if np in ['body.weight', 'body.bias', 'weight', 'bias']:  # weight is scale, bias is shift
                        yield p  # params.append(p)
        # return params
    elif config.specific_param == 'bias':
        params = []
        for nm, m in model.named_modules():
            for np, p in m.named_parameters():
                if 'bias' in np:  # weight is scale, bias is shift
                    yield p  # params.append(p)

        # return params
    elif config.specific_param == 'full':
        params = []
        for nm, m in model.named_modules():
            for np, p in m.named_parameters():
                yield p  # params.append(p)

    elif config.specific_param == 'LN_enc':
        params = []
        for nm, m in model.named_modules():
            try:
                if nm.split('.')[1] == 'encoder':
                    if isLN(m):
                        for np, p in m.named_parameters():
                            if np in ['body.weight', 'body.bias', 'weight', 'bias']:  # weight is scale, bias is shift
                                yield p  # params.append(p)
            except:
                pass
            elif config.specific_param == 'LN_others':
            params = []
            for nm, m in model.named_modules():
                try:
                    if nm.split('.')[1] != 'encoder':
                        if isLN(m):
                            for np, p in m.named_parameters():
                                if np in ['body.weight', 'body.bias', 'weight',
                                          'bias']:  # weight is scale, bias is shift
                                    yield p  # params.append(p)
                except:
                    pass

            # return params
        elif config.specific_param == 'bias_enc':
        params = []
        for nm, m in model.named_modules():
            try:
                if nm.split('.')[1] == 'encoder':
                    for np, p in m.named_parameters():
                        if 'bias' in np:  # weight is scale, bias is shift
                            yield p  # params.append(p)
            except:
                pass

    elif config.specific_param == 'bias_others':
        params = []
        for nm, m in model.named_modules():
            try:
                if nm.split('.')[1] != 'encoder':
                    for np, p in m.named_parameters():
                        if 'bias' in np:  # weight is scale, bias is shift
                            yield p  # params.append(p)
            except:
                pass

    # return params
    elif config.specific_param == 'vanilla_enc':
        params = []
        for nm, m in model.named_modules():
            if nm.split('.')[0] == 'encoder':
                for np, p in m.named_parameters():
                    yield p  # params.append(p)

    elif config.specific_param == 'LN_enc_others':
        params = []
        for nm, m in model.named_modules():
            try:
                if nm.split('.')[1] == 'encoder':
                    if isLN(m):
                        for np, p in m.named_parameters():
                            if np in ['body.weight', 'body.bias', 'weight', 'bias']:  # weight is scale, bias is shift
                                yield p  # params.append(p)
                else:
                    for np, p in m.named_parameters():
                        yield p
            except:
                pass

    elif config.specific_param == 'Bias_in_LN_enc_others':
        params = []
        for nm, m in model.named_modules():
            try:
                if nm.split('.')[1] == 'encoder':
                    if isLN(m):
                        for np, p in m.named_parameters():
                            if np in ['body.bias', 'bias']:  # weight is scale, bias is shift
                                yield p  # params.append(p)
                else:
                    for np, p in m.named_parameters():
                        yield p
            except:
                pass
    elif config.specific_param == 'except_enc':
        params = []
        for nm, m in model.named_modules():
            try:
                if nm.split('.')[1] == 'encoder':
                    pass
                else:
                    for np, p in m.named_parameters():
                        yield p
            except:
                pass
    elif config.specific_param == 'except_mm_bias':
        params = []
        for nm, m in model.named_modules():
            try:
                if nm.split('.')[1] == 'matching_module':
                    pass
                else:
                    for np, p in m.named_parameters():
                        if np in ['body.bias', 'bias']:  # weight is scale, bias is shift
                            yield p  # params.append(p)
            except:
                pass
    elif config.specific_param == 'mm':
        params = []
        for nm, m in model.named_modules():
            try:
                if nm.split('.')[1] == 'matching_module':
                    for np, p in m.named_parameters():
                        yield p
            except:
                pass
