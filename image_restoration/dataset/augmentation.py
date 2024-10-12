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
    cut_rat = np.sqrt(lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
