from tools.visualize import visualize_tsne, visualize_tsne_raw
import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
from torch.utils import data
import numpy as np
import random
from tqdm import tqdm
from config import get_config
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from typing import Tuple, List, Union

import torchvision.transforms as T
from easydict import EasyDict
from scipy.interpolate import make_interp_spline, BSpline

def load_embedding(path, label=False, stack=True, key='embedding'):
    data = torch.load(path)
    emb = data[key] if key else data
    if stack:
        if isinstance(emb[0], torch.Tensor):
            emb = torch.stack(emb, dim=0).cpu().squeeze()
        else:
            emb = np.stack(emb, axis=0).squeeze()
    # emb = list(map(lambda x: x.cpu()[0], emb))
    if label:
        return emb, data['labels']
    return emb  

def get_embedding(src_embeddings: List[torch.Tensor]):
    src_embeddings = torch.stack(src_embeddings).to('cpu')
    img_emb_src = src_embeddings.mean(dim=0, keepdim=True)
    img_emb_src /= img_emb_src.norm(dim=-1, keepdim=True)
    img_emb_src = img_emb_src.repeat(1,1).type(torch.float32)  # (B,1024)
    print(img_emb_src.shape)
    return img_emb_src.cpu()


def norm_embeddings(src_embeddings):
    src_embeddings = src_embeddings.clone()
    for i in range(len(src_embeddings)):
        src_embeddings[i] /= src_embeddings[i].norm(dim=-1, keepdim=True)
    return torch.stack(src_embeddings)

def normalize(embeddings): #list of list of embds
    results = []
    for lst in embeddings:
        res = []
        for emb in lst:
            emb = emb.clone()
            emb /= emb.norm(dim=-1, keepdim=True)
            res.append(emb)
        results.append(res)
    return results


def interpolated_array(x: np.ndarray, y: np.ndarray, num:int=100):
    xx = np.linspace(x.min(), x.max(), num)
    spl = make_interp_spline(x, y, k=3)
    return xx, spl(xx)

def interpolated(df: pd.DataFrame, col: str, num:int=100):
    if type(col) == list:
        arr = [interpolated_array(df.index, df[c], num) for c in col]
        return arr[0][0], [a[1] for a in arr]
    return interpolated_array(df.index, df[col], num)
