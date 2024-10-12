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


class MetaWeather(nn.Module):
    def __init__(self, config, swin_config, n_tasks, 
                 num_decoders=[1, 1, 1, 1], dim=[1024, 1024, 512, 256], 
                 mm_settings={}):
        super(MetaWeather, self).__init__()
        if config.enc == 'swin':
            self.encoder = build_swin_modularized(swin_config, n_tasks)
            self.logger = create_logger(output_dir=swin_config.OUTPUT, name=f"{swin_config.MODEL.NAME}")
            load_pretrained(swin_config, self.encoder, self.logger, n_tasks)
            self.decoder = MetaWeatherDecoder(num_decoders=num_decoders, dim=dim, mode=config.enc)
            self.matching_module = SpChMatchingModule(**mm_settings)
            self.project_out = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1, bias=True)    
            
            self.refine_naf = NAFBlock(4)
            self.refine_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1, bias=True)   

        else :
            raise NotImplementedError(f'Invalid encoder \'{config.enc}\'')


        print(f'MetaWeather initialized: encoder={config.enc}, n_tasks={n_tasks}')   



    def forward(self, Xs: torch.Tensor, Ys: torch.Tensor, Xq: torch.Tensor, t_idx: torch.Tensor = None):
        Bs, Ts, Ns = Xs.shape[:3]
        Bq, Tq, Nq = Xq.shape[:3]

        Xs = from_6d_to_4d(Xs, contiguous=True)
        Ys = from_6d_to_4d(Ys, contiguous=True)
        Xq = from_6d_to_4d(Xq, contiguous=True)
        
        Ys = Xs - Ys # deg pattern

        t_idx_q = repeat(t_idx, 'B T -> (B T N)', N=Nq) if t_idx is not None else None
        t_idx_s = repeat(t_idx, 'B T -> (B T N)', N=Ns) if t_idx is not None else None

        eXs = self.encoder(Xs, t_idx_s) #down -> up, C=[1024 1024 512 256]
        eYs = self.encoder(Ys, t_idx_s)
        eXq = self.encoder(Xq, t_idx_q)

        eXs = map_fn(from_4d_to_6d, *eXs, B=Bs, T=Ts, N=Ns)
        eYs = map_fn(from_4d_to_6d, *eYs, B=Bs, T=Ts, N=Ns)
        eXq = map_fn(from_4d_to_6d, *eXq, B=Bq, T=Tq, N=Nq)

        matched = self.matching_module(eXq, eXs, eYs)
        matched = from_6d_to_4d(matched)

        decoded = self.decoder(matched)
        # decoded = self.decoder(eXq)
        output = self.refine_naf(decoded)
        output = self.project_out(output)
        # output = self.project_out(decoded)

        output = Xq - output # residual
        output = self.refine_conv(output)
        # output = self.refine_final(output)
        output = from_4d_to_6d(output, B=Bq, T=Tq, N=Nq)        
        return output