import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Callable, Dict, List
from .base import IRUnitDataset

from .augmentation import CutmixMulti

import subprocess
import time
import os
import random


class IRDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 datasets: List[IRUnitDataset], 
                 shot: int, 
                 dset_size: int = None,
                 precision: str = 'fp32',
                 binary_augmentation: bool = True) -> None:
        super().__init__()
        assert shot > 0
        assert len(datasets) > 0

        self.datasets = datasets
        self.dataset_lengths = list(map(len, datasets))
        self.tasks_per_batch = len(self.datasets)
        self.shot = shot
        self.precision = precision
        self.binary_augmentation = CutmixMulti() if binary_augmentation else None
        self.max_bin_aug = 2
        self.dset_size = dset_size if dset_size is not None else (min(self.dataset_lengths) // shot)


    # choose where to sample data
    def sample_tasks(self) -> np.ndarray:
        replace = len(self.datasets) < self.tasks_per_batch
        return np.random.choice(len(self.datasets), self.tasks_per_batch, replace=replace)
        
    # (N, C, H, W)
    def sample_data(self, task: int) -> Tuple[Tensor, Tensor]:
        dataset = self.datasets[task]
        idx = np.random.choice(len(dataset), self.shot * 2, replace=False)

        data = [dataset[i] for i in idx] # im, gt, id

        imgs, gts = zip(*data)

        imgs = torch.stack(imgs)
        gts = torch.stack(gts)

        #preprocess?
        # augmentation

        return imgs, gts


    # (T, N, C, H, W)
    def __getitem__(self, _) -> Tuple[Tensor, Tensor, Tensor]:
        tasks = self.sample_tasks() #(N)
        if self.binary_augmentation is not None:
            tasks_aug = np.array([self.sample_tasks() for _ in range(self.max_bin_aug)])

        imgs = []
        gts = []
        for i, task in enumerate(tasks):
            X_, Y_ = self.sample_data(task) # (N, C, H, W)

            if self.binary_augmentation is not None:
                aug_num = np.random.choice(self.max_bin_aug)
                if aug_num > 0:
                    aug_data = [self.sample_data(t) for t in tasks_aug[:aug_num,i]]
                    #X_aug, Y_aug = self.sample_data(tasks_aug[i])
                    X_aug, Y_aug = zip(*aug_data)
                    X_, Y_ = self.binary_augmentation([X_] + list(X_aug), [Y_] + list(Y_aug))

            imgs.append(X_)
            gts.append(Y_)

        X = torch.stack(imgs)
        Y = torch.stack(gts)

        if self.precision == 'fp16':
            X = X.half()
            Y = Y.half()
        elif self.precision == 'bf16':
            X = X.bfloat16()
            Y = Y.bfloat16()

        return X, Y, torch.tensor(tasks)

    
    def __len__(self) -> int:
        return self.dset_size