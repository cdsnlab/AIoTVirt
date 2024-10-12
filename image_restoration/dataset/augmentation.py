import random
import math
import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur


def normalize(x):
    if x.max() == x.min():
        return x - x.min()
    else:
        return (x - x.min()) / (x.max() - x.min())


def linear_sample(p_range):
    if isinstance(p_range, float):
        return p_range
    else:
        return p_range[0] + random.random()*(p_range[1] - p_range[0])
    
    
def log_sample(p_range):
    if isinstance(p_range, float):
        return p_range
    else:
        return math.exp(math.log(p_range[0]) + random.random()*(math.log(p_range[1]) - math.log(p_range[0])))
    
    
def categorical_sample(p_range):
    if isinstance(p_range, (float, int)):
        return p_range
    else:
        return p_range[np.random.randint(len(p_range))]
    
    
def rand_bbox(size, lam):
    H, W = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Augmentation:
    pass


class RandomHorizontalFlip(Augmentation):
    def __init__(self):
        self.augmentation = lambda x: torch.flip(x, dims=[-1])
        
    def __str__(self):
        return 'RandomHorizontalFlip Augmentation'
        
    def __call__(self, *arrays, get_augs=False):
        if random.random() < 0.5:
            if len(arrays) == 1:
                if get_augs:
                    return self.augmentation(arrays[0]), self.augmentation
                else:
                    return self.augmentation(arrays[0])
            else:
                arrays_flipped = []
                for array in arrays:
                    arrays_flipped.append(self.augmentation(array))
                if get_augs:
                    return arrays_flipped, self.augmentation
                else:
                    return arrays_flipped
        else:
            if len(arrays) == 1:
                if get_augs:
                    return arrays[0], lambda x: x
                else:
                    return arrays[0]
            else:
                if get_augs:
                    return arrays, lambda x: x
                else:
                    return arrays
    
