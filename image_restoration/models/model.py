import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import List
from .matching_modules import SpChMatchingModule

from .NAFNet.NAFNet_arch import NAFBlock, NAFNetEnc
from .vit.swin_modularized import build_swin_modularized
from .vit.config import get_config
from .vit.utils import load_pretrained
from .vit.logger import create_logger

from .reshape import from_6d_to_4d, from_4d_to_6d
from .utils import map_fn


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class MetaWeatherDecoderBlock(nn.Module):
    def __init__(self, num_decoder: int, dim: int, upsample: bool=True):
        super(MetaWeatherDecoderBlock, self).__init__()
        self.decoder = nn.Sequential(
            *[NAFBlock(dim) for _ in range(num_decoder)] 
        )
        self.up = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=False),
            nn.PixelShuffle(2)
        ) if upsample else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor=None):
        if skip is not None:
            x = x + skip

        x = self.decoder(x)

        if self.up is not None:
            x = self.up(x)
        
        return x

class MetaWeatherDecoder(nn.Module):
    def __init__(self, num_decoders: List=[1, 1, 1, 1], dim=[1024, 1024, 512, 256], mode='swin'):
        super(MetaWeatherDecoder, self).__init__()

        self.num_decoders = num_decoders
        self.num_levels = len(self.num_decoders)
        self.dim = dim
        
        if mode == 'swin':
            upsample_from = 1
            self.last_up = nn.PixelShuffle(8)
        elif mode == 'nafnet':
            upsample_from = 0
            self.last_up = nn.PixelShuffle(2)

        self.blocks = nn.ModuleList([
            MetaWeatherDecoderBlock(n, d, upsample=upsample_from <= i < self.num_levels-1) for i, (n, d) in enumerate(zip(num_decoders, dim))
        ])
        

    def forward(self, encs: List[torch.Tensor]): #down -> up
        x = encs[0]
        for i, (enc, block) in enumerate(zip(encs, self.blocks)):
            x = block(x, enc if i>0 else None)
        
        x = self.last_up(x)
        return x
