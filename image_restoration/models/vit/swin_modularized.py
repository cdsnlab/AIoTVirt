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
