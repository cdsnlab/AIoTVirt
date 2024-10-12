import torch
from PIL import Image
import random
from torchvision.transforms import Compose, ToTensor, Normalize
import re
import os
import numpy as np
from typing import Tuple, Callable, Dict

import albumentations

import time
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
        """
            * mode: random_crop, center_crop, resize, none
        """
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
        
        if self.mode in ['random_crop', 'center_crop', 'resize']:
            assert crop_size is not None
            augmentations_map = {
                'random_crop': albumentations.RandomCrop,
                'center_crop': albumentations.CenterCrop,
                'resize': albumentations.Resize
            }
            augs = [augmentations_map[self.mode](width=self.crop_size[0], height=self.crop_size[1])]
        else:
            augs = []

        if image_augmentation:
            augs.append(albumentations.HorizontalFlip(p=0.5))

        self.image_augmentation = albumentations.Compose(augs, additional_targets={'gt': 'image'}) if augs else None

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

    def get_images(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if index >= len(self.input_names):
            index = np.random.choice(len(self.input_names))

        if self.cache is not None and index in self.cache:
            return self.cache[index]

        input_name, gt_name = self.input_names[index], self.gt_names[index]
        img_id = os.path.splitext(os.path.basename(input_name))[0]

        input_img = Image.open(os.path.join(self.train_data_dir, input_name)).convert("RGB")
        gt_img = Image.open(os.path.join(self.train_data_dir, gt_name)).convert("RGB")

        input_img, gt_img = self._resize_if_needed(input_img, gt_img)
        input_im, gt = self._apply_augmentation(input_img, gt_img)

        if self.cache is not None:
            self.cache[index] = (input_im, gt, img_id)

        return input_im, gt, img_id if self.return_image_id else (input_im, gt)

    # Add helper methods for resizing and augmentation
    def _resize_if_needed(self, input_img, gt_img):
        width, height = input_img.size
        if self.crop_size and (width < self.crop_size[0] or height < self.crop_size[1]):
            new_size = (max(width, self.crop_size[0]), max(height, self.crop_size[1]))
            input_img = input_img.resize(new_size, Image.ANTIALIAS)
            gt_img = gt_img.resize(new_size, Image.ANTIALIAS)
        return input_img, gt_img

    def _apply_augmentation(self, input_img, gt_img):
        if self.image_augmentation is not None:
            augmented = self.image_augmentation(image=np.array(input_img), gt=np.array(gt_img))
            input_im = self.transform_input(augmented['image'])
            gt = self.transform_gt(augmented['gt'])
        else:
            input_im = self.transform_input(input_img)
            gt = self.transform_gt(gt_img)
        return input_im, gt



    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self) -> int:
        return self.dset_size
    