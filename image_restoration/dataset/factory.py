from typing import Tuple, Callable, Dict

from .base import IRUnitDataset
from .irdata import IRDataset, IRFinetuneDataset, IREvalDataset

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import os
from typing import List
from einops import rearrange
from .utils import TASK_DATASETS_TEST, TASK_DATASETS_TRAIN

"""
Rain13k : same name
snow100k: same name
Rain1400:
    input 901_1.jpg / gt 901.jpg
Outdoor-Rain: 
    input im_0001_s80_a04.png / gt im_0001.png
Raindrop:
    input 0_rain.png / gt 0_clean.png
"""


def _by_basename(fn_gt_name: Callable[[str], str]) -> Callable[[str], str]:
    def _get_gt(input_name: str) -> str:
        dir, base = os.path.split(input_name)
        gt_base = fn_gt_name(base)
        return os.path.join(dir, gt_base)
    return _get_gt

def _get_gt_rain1400(input_name: str) -> str:
    dir, base = os.path.split(input_name)
    gt_base = '_'.join(base.split('_')[:-1]) + '.jpg'
    return os.path.join(dir, gt_base)

def _get_gt_outdoor_rain(input_name: str) -> str:
    dir, base = os.path.split(input_name)
    gt_base = '_'.join(base.split('_')[:2]) + '.png'
    return os.path.join(dir, gt_base)


def _get_gt_raindrop(input_name: str) -> str:
    dir, base = os.path.split(input_name)
    gt_base = base.replace('_rain', '_clean')
    return os.path.join(dir, gt_base)


def _get_gt_cityscape(input_name: str) -> str:
    dir, base = os.path.split(input_name)
    base = '.'.join(base.split('.')[:-1])
    gt_base = '_'.join(base.split('_')[:4]) + '.png'
    return os.path.join(dir, gt_base)


def _get_gt_ots(input_name: str) -> str:
    dir, base = os.path.split(input_name)
    extend = '.png' if 'test' in dir else '.jpg'
    return os.path.join(os.path.dirname(dir), base.split('_')[0] + extend)


def _get_gt_bid(input_name: str) -> str:
    input_name = os.path.join('gt', input_name.split('/')[-1])
    return input_name

def _get_gt_gtrain(input_name: str) -> str:
    dir, base = os.path.split(input_name)
    base = '.'.join(base.split('.')[:-1])
    gt_base = '-'.join(base.split('-')[:-2]) + '-C-000.png'
    return os.path.join(dir, gt_base)

def _get_gt_spadata(input_name: str) -> str:
    dir, base = os.path.split(input_name)
    gt_base = base.split('.')[0] + 'gt.png'
    return os.path.join(dir, gt_base).replace('rain/', 'gt/')

def _get_gt_ohaze(input_name: str) -> str:
    dir, base = os.path.split(input_name)
    gt_base = base.replace('_hazy', '_GT')
    return os.path.join(dir, gt_base)

gt_name_functions: Dict[str, Callable[[str], str]] = {
    'rain1400': _get_gt_rain1400,
    'outdoor-rain': _get_gt_outdoor_rain,
    'raindrop': _get_gt_raindrop,
    'fog': _get_gt_cityscape,
    'ots': _get_gt_ots,
    'case1': _get_gt_bid,
    'case2': _get_gt_bid,
    'case3': _get_gt_bid,
    'case4': _get_gt_bid,
    'case5': _get_gt_bid,
    'case6': _get_gt_bid,
    'gtrain': _get_gt_gtrain,
    'spadata': _get_gt_spadata,
    'ohaze': _get_gt_ohaze,
}