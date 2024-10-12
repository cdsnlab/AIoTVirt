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
