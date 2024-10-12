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
    
    
class IRFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset: IRUnitDataset, 
                 shot: int, 
                 support_idx: int,
                 dset_size: int=1,
                 shuffle_idx: bool = False,
                 shuffle: bool = True,
                 precision: str = 'fp32',
                 meta_info_path: str = 'data/meta_info') -> None:
        super().__init__()
        assert shot > 0

        self.dataset = dataset
        self.shot = shot
        self.precision = precision
        self.support_idx = support_idx
        self.dset_size = dset_size
        self.shuffle_idx = shuffle_idx
        self.shuffle = shuffle
        self.meta_info_path = os.path.join(meta_info_path, dataset.name)
        self.offset = support_idx * shot

        perm_path = os.path.join(self.meta_info_path, 'idxs_perm_finetune.pth')
        if not os.path.exists(self.meta_info_path):
            subprocess.check_output(['mkdir', '-p', self.meta_info_path])
        

        if not os.path.exists(perm_path):
            idxs_perm = torch.randperm(len(dataset))
            torch.save(idxs_perm, perm_path)
        else:
            idxs_perm = torch.load(perm_path)
        self.idxs_perm = idxs_perm
    

    # (T, N, C, H, W)
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        idxs = [((idx+i) % self.shot) + self.offset for i in range(self.shot)]
        if self.shuffle:
            random.shuffle(idxs)

        imgs = []
        gts = []
        for i in idxs:
            if self.shuffle_idx:
                i = self.idxs_perm[i % len(self.dataset)]
            X_, Y_ = self.dataset[i] # (C, H, W)

            imgs.append(X_)
            gts.append(Y_)

        X = torch.stack(imgs).unsqueeze(0)
        Y = torch.stack(gts).unsqueeze(0)

        if self.precision == 'fp16':
            X = X.half()
            Y = Y.half()
        elif self.precision == 'bf16':
            X = X.bfloat16()
            Y = Y.bfloat16()

        return X, Y

    
    def __len__(self) -> int:
        return self.dset_size
    
    
class IREvalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset: IRUnitDataset, 
                 precision: str = 'fp32') -> None:
        super().__init__()

        self.dataset = dataset
        self.precision = precision

    # (T, N, C, H, W)
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        X_, Y_ = self.dataset[idx] # (C, H, W)
        
        X = X_[None,None,:]
        Y = Y_[None,None,:]

        if self.precision == 'fp16':
            X = X.half()
            Y = Y.half()
        elif self.precision == 'bf16':
            X = X.bfloat16()
            Y = Y.bfloat16()

        return X, Y

    
    def __len__(self) -> int:
        return len(self.dataset)