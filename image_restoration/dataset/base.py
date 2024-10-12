import torch
from PIL import Image
import random
from torchvision.transforms import Compose, ToTensor, Normalize
import re
import os
import numpy as np
from typing import Tuple, Callable, Dict

import albumentations

class IRUnitDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 crop_size: Tuple[int, int], 
                 data_dir: str, 
                 data_filename: str, 
                 name: str=None, 
                 fn_gt_name: Callable[[str], str] = None, 
                 shuffle: bool = True, 
                 dset_size: int = None,
                 image_augmentation: bool = False,
                 return_image_id: bool = False,
                 mode: str = 'random_crop',
                 cache: str = None):
        super().__init__()
        self.fn_gt_name = fn_gt_name
        self.name = name

        with open(data_filename) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            if shuffle:
                random.shuffle(input_names)

        self.input_names = input_names
        self.gt_names = [self._get_gt_name(n) for n in input_names]
        self.crop_size = crop_size
        self.train_data_dir = data_dir
        self.dset_size = dset_size if dset_size is not None else len(input_names)
        self.to_shuffle = shuffle
        self.return_image_id = return_image_id
        self.mode = mode
        self.cache_mode = cache

        self.transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_gt = Compose([ToTensor()])

        assert mode in ['random_crop', 'center_crop', 'resize', 'none']

        if mode in ['random_crop', 'center_crop', 'resize']:
            assert crop_size is not None

        if self.mode == 'random_crop':
            augs = [albumentations.RandomCrop(width=self.crop_size[1], height=self.crop_size[0])]
        elif self.mode == 'center_crop':
            augs = [albumentations.CenterCrop(width=self.crop_size[0], height=self.crop_size[1])]
        elif self.mode == 'resize':
            augs = [albumentations.Resize(height=self.crop_size[1], width=self.crop_size[0])]
        else:
            augs = []

        if image_augmentation:
            augs += [albumentations.HorizontalFlip(p=0.5)]

        if len(augs) > 0:
            self.image_augmentation = albumentations.Compose(augs, additional_targets={'gt': 'image'})
        else:
            self.image_augmentation = None
            
        if self.cache_mode is None:
            self.cache = None
        elif self.cache_mode == 'mem':
            self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, str]] = dict()
        else:
            raise NotImplementedError(f'Specified cache mode {cache} is not implemented')
        
    
    def _get_gt_name(self, input_name: str) -> str:
        gt_name = input_name.strip().replace('input/', 'gt/')
        gt_name = self.fn_gt_name(gt_name) if self.fn_gt_name is not None else gt_name
        if input_name == gt_name:
            raise NotImplementedError(f'Error: input and gt names are the same. Did you implement the gt name converter function correctly?')
        return gt_name