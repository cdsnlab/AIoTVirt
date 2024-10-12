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

def create_unit_dataset(config, task: str, split: str, mode: str, dset_size: int = None, image_augmentation: bool = True, shuffle: bool=True, cache: str=None) -> IRUnitDataset:
    if task not in config.datasets:
        raise Exception(f'task \'{task}\' is not in config')

    dconf = config.datasets[task]
    name = task.strip().lower()
    return IRUnitDataset(crop_size=(config.img_size, config.img_size), 
                         data_dir=os.path.join(config.data_root, dconf.path), 
                         data_filename=os.path.join(config.data_root, dconf[split]), 
                         mode=mode,
                         name=name,
                         shuffle=shuffle,
                         dset_size=dset_size,
                         image_augmentation=image_augmentation,
                         fn_gt_name=gt_name_functions[name] if name in gt_name_functions else None,
                         cache=cache)
    

def create_dataset(config, split: str, mode: str, tasks: List[str]=None, dset_size: int = None, verbose=True) -> IRDataset:
    """
        split: train, val, or test
    """
    if tasks is None:
        tasks = config.datasets.keys()
    datasets = [create_unit_dataset(config, task, split=split, dset_size=None, image_augmentation=config.image_augmentation, mode=mode)
                for task, dconf in config.datasets.items() if (split in dconf) and (task in tasks)]
    
    if verbose:
        print(f'{len(datasets)} datasets found: ' + ', '.join([f'{ds.name}[{len(ds)}]' for ds in datasets]))

    return IRDataset(
        datasets=datasets,
        shot=config.shot,
        dset_size=dset_size,
        precision=config.precision,
        binary_augmentation=config.binary_augmentation
    )