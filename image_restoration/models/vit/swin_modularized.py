# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import repeat, rearrange
from functools import partial

class LinearModularized(nn.Linear):
    def __init__(self, n_tasks=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.n_tasks = n_tasks
        if self.n_tasks > 0:
            assert self.bias is not None
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_tasks).contiguous())

    def forward(self, input, t_idx=None):
        if self.n_tasks > 0:
            assert t_idx is not None
            output = F.linear(input, self.weight, None)
            return output + self.bias[t_idx][:, None]
        else:
            return F.linear(input, self.weight, self.bias)


class MlpModularized(nn.Module):
    def __init__(self, n_tasks, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LinearModularized(n_tasks, in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = LinearModularized(n_tasks, hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, t_idx=None):
        x = self.fc1(x, t_idx)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, t_idx)
        x = self.drop(x)
        return x


class LayerNormModularized(nn.LayerNorm):
    def __init__(self, n_tasks=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.n_tasks = n_tasks
        if self.n_tasks > 0:
            assert self.elementwise_affine
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_tasks).contiguous())

    def forward(self, input, t_idx=None):
        if self.n_tasks > 0:
            assert t_idx is not None
            output = F.layer_norm(input, self.normalized_shape, self.weight, None, self.eps)
            return output + self.bias[t_idx][:, None]
        else:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
