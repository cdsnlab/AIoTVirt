import torch
import numpy as np

#'0: rain, 1: fog, 2: snow, 3: raindrop, 4: outdoorrain'
TASK_DATASETS_TRAIN = ['rain13k', 'fog', 'snow100k', 'raindrop']
TASK_DATASETS_TEST = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'reside_urhi', 'gtrain', 'ohaze', 'spadata', 'realsnow', 'rainds_raindrop', 'rainds_rainstreak', 'rainds_rainstreak_drop']

def to_device(data, device=None, dtype=None):
    '''
    Load data with arbitrary structure on device.
    '''
    def to_device_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)
        elif isinstance(data, tuple):
            return tuple(map(to_device_wrapper, data))
        elif isinstance(data, list):
            return list(map(to_device_wrapper, data))
        elif isinstance(data, dict):
            return {key: to_device_wrapper(data[key]) for key in data}
        else:
            return data
            
    return to_device_wrapper(data)