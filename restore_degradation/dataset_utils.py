from typing import Tuple, Callable, Dict

from torch.utils.data import DataLoader
import yaml
from easydict import EasyDict

import os

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
    return os.path.join(os.path.dirname(dir), base.split('_')[0] + '.jpg')

def _get_gt_default(input_name: str) -> str:
    return input_name

gt_name_functions: Dict[str, Callable[[str], str]] = {
    'rain1400': _get_gt_rain1400,
    'outdoor-rain': _get_gt_outdoor_rain,
    'raindrop': _get_gt_raindrop,
    'cityscape': _get_gt_cityscape,
    'ots': _get_gt_ots,
    'rain13k': _get_gt_default,
    'snow100k': _get_gt_default
}

def get_gt_name(dataset: str, input_name: str) -> str:
    if dataset in gt_name_functions:
        return gt_name_functions[dataset](input_name)
    else:
        return input_name

def load_data():
    config = EasyDict()
    with open('data.yaml') as f:
        data_conf = yaml.safe_load(f)
        config.data_root = data_conf['data_root']
        config.datasets = data_conf['datasets']
    return config


def get_dataset(split: str = 'train', task: str = None):
    config = load_data()

    input_names = []
    gt_names = []

    for dset, elem in config.datasets.items():
        if task is not None and task != dset: continue

        if split in elem:
            train_list = os.path.join(config.data_root, elem[split])
            data_path = os.path.join(config.data_root, elem['path'])
        
            with open(train_list) as f:
                contents = f.readlines()
                inputs = [os.path.join(data_path, i.strip()) for i in contents]
                gts = [get_gt_name(dset, i.strip().replace('input/','gt/')) for i in inputs]

                input_names += inputs
                gt_names += gts
    return input_names, gt_names