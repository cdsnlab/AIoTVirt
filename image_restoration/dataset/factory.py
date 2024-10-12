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
    
def get_train_tasks(config):
    tasks = TASK_DATASETS_TRAIN.copy()
    return tasks

def get_train_dataloader(config, split: str='train', pin_memory=True, verbose=True):
    # if config.no_eval:
    dset_size = config.n_steps*config.batch_size
    # else:
    # dset_size = config.val_iter*config.batch_size
    # import pdb; pdb.set_trace()
    tasks = get_train_tasks(config)
    dataset = create_dataset(config, split=split, tasks=tasks, mode='random_crop', dset_size=dset_size, verbose=verbose)
    # --- Distributed Data Parallel initialize --- #
    if config.ddp is True:    
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
        loader = DataLoader(dataset, batch_size=config.batch_size // torch.cuda.device_count(), shuffle=False, pin_memory=pin_memory, drop_last=True, num_workers=config.num_workers, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=config.batch_size // torch.cuda.device_count(), shuffle=False, pin_memory=pin_memory, drop_last=True, num_workers=config.num_workers)
    loader = DataLoader(dataset, batch_size=config.batch_size // torch.cuda.device_count(), shuffle=False, pin_memory=pin_memory, drop_last=True, num_workers=config.num_workers)
    if verbose:
        print(f'DataLoader[{len(loader)}] with bs={loader.batch_size}')
    return loader


def get_finetune_dataloader(config, task: str, mode: str='resize', split: str='train', support_idx: int=0, pin_memory=True, cache: str='mem', verbose=True):
    # if config.no_eval:
    dset_size = config.n_steps*config.batch_size
    # else:
    # dset_size = config.val_iter*config.batch_size
    dset = create_unit_dataset(config, task=task, split=split, mode=mode, dset_size=config.shot, image_augmentation=False, shuffle=False, cache=cache)
    dataset = IRFinetuneDataset(dset, shot=config.shot, support_idx=support_idx, dset_size=dset_size, precision=config.precision, shuffle_idx=False)
    # --- Distributed Data Parallel initialize --- #
    if config.ddp is True:
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
        loader = DataLoader(dataset, batch_size=config.batch_size // torch.cuda.device_count(), shuffle=False, pin_memory=pin_memory, drop_last=True, num_workers=config.num_workers, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=config.batch_size // torch.cuda.device_count(), shuffle=False, pin_memory=pin_memory, drop_last=True, num_workers=config.num_workers)
    return loader

def get_eval_dataloader(config, task: str, mode: str='resize', split: str='val', pin_memory=True, verbose=True):
    assert task in config.datasets

    dset = create_unit_dataset(config, task=task, split=split, mode=mode, dset_size=None, image_augmentation=False)
    dataset = IREvalDataset(dset, precision=config.precision)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=pin_memory, drop_last=True, num_workers=config.num_workers)
    if verbose:
        print(f'DataLoader(val, {task})[{len(loader)}]')
    return loader

def get_val_dataloaders(config, task: str=None, support_data=None, pin_memory=True, verbose=True):
    tasks = TASK_DATASETS_TRAIN.copy()
    # tasks = config.datasets.keys() if task is None else [task]
    tag = 'mtrain_valid' if task is None else 'mtest_valid'

    if config.stage == 0:
        if config.ddp is True and dist.get_rank() != 0:
            dataloaders = None
        else:
            dataloaders = {
                        task_name: get_eval_dataloader(config, task_name, split='val', mode='center_crop', pin_memory=pin_memory, verbose=verbose)
                        for task_name in tasks if 'val' in config.datasets[task_name]
                }
        return dataloaders, tag
    else:
        class SubQueryDataset:
            def __init__(self, data):
                X, Y = data
                self.X = X #(1, 1, N, C, H, W)
                self.Y = Y
                self.n_query = self.X.shape[2] // 2  # N/2
            
            def __len__(self):
                return self.n_query
            
            def __getitem__(self, idx):
                # import pdb; pdb.set_trace()
                return (self.X[0, 0, self.n_query+idx][None, None, :],
                        self.Y[0, 0, self.n_query+idx][None, None, :])
                #(1, 1, C, H, W)
                
        dset = SubQueryDataset(support_data) #X, Y
        
        return torch.utils.data.DataLoader(dset, shuffle=False, batch_size=len(dset))
    
# for meta-test validation
def get_support_data(config, task: str, split: str='shots', support_idx: int=0, pin_memory=True, verbose=True):
    dset = create_unit_dataset(config, task=task, split=split, mode='resize', dset_size=config.shot, image_augmentation=False, shuffle=False)
    dataset = IRFinetuneDataset(dset, shot=config.shot, support_idx=support_idx, dset_size=config.shot, precision=config.precision, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for support_data in loader: break
    return support_data

# for train
def generate_support_data(config, data_path: str, split: str='shots', support_idx: int=0, verbose:bool=True):
    if os.path.exists(data_path):
        support_data = torch.load(data_path)
    else:
        support_data = {}

    modified = False
    tasks = list(filter(lambda x: split in config.datasets[x], config.datasets.keys()))
    print(f'Tasks: {tasks}')

    for task in tasks:
        if task in support_data: continue

        dataset = create_unit_dataset(config, task=task, split=split, mode='center_crop', dset_size=None, image_augmentation=False)

        dloader = DataLoader(dataset, batch_size=config.shot, shuffle=False, num_workers=0)
        for idx, batch in enumerate(dloader):
            if idx == support_idx: break
        
        X, Y = batch  #N, C, H, W
        support_data[task] = (X[None, None, :], Y[None, None, :])

        if verbose:
            print(f'generated support data for task {task}')
        modified = True
    
    if modified:
        torch.save(support_data, data_path)
    
    return support_data