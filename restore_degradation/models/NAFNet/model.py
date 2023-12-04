# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .NAFNet_arch import  NAFBlock

from einops import rearrange
from models.Restormer.reshape import from_4d_to_6d, from_6d_to_4d, parse_BTN
from models.Restormer.Restormer import grad_reverse
from models.matching_modules import ChannelMatchingModule, BasicCosineMatchingModule
from typing import List


class NAFNet_Encoder(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

    def forward(self, inp):
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        return *encs, x


class NAFNet_Decoder(nn.Module):
    def __init__(self, img_channel=3, width=16, dec_blk_nums=[], IB=False):
        super().__init__()

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.ib=IB

        chan = width * (2 ** len(dec_blk_nums))

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            
    def forward(self, *args, residual=None):
        x, *encs = args[::-1]
        if residual is not None:
            r, *residual = residual[::-1]
            x = x + r
        else:
            residual = [None] * len(self.decoders)

        for idx, (decoder, up, enc_skip, res) in enumerate(zip(self.decoders, self.ups, encs, residual)):
            if idx==0 and self.ib:
                x = up(grad_reverse(x))
            else:
                x = up(x)
            x = x + enc_skip

            if res is not None:
                x = x + res

            x = decoder(x)

        x = self.ending(x)
        return x



class NAFNet_Fewshot(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1, 
                 enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1], num_heads=[1,2,4,8,8],
                 use_residual: bool = False):
        super().__init__()

        self.encoder = NAFNet_Encoder(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums)
        self.encoder_label = NAFNet_Encoder(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums)
        self.decoder = NAFNet_Decoder(img_channel=img_channel, width=width, dec_blk_nums=dec_blk_nums)
        # self.decoder_deg_Q = NAFNet_Decoder(img_channel=img_channel, width=width, dec_blk_nums=dec_blk_nums, IB=True)
        # self.decoder_deg_S = NAFNet_Decoder(img_channel=img_channel, width=width, dec_blk_nums=dec_blk_nums)
        # self.matching_module = ChannelMatchingModule(dim=width, num_heads=num_heads, n_levels = len(enc_blk_nums) + 1)
        self.matching_module = BasicCosineMatchingModule()
        self.use_residual = use_residual

        self.padder_size = 2 ** len(enc_blk_nums)

    def reshape_encoding(self, xs: List[torch.Tensor], B, T, N):
        def _reshape_encoding(x):
            return from_4d_to_6d(x, B=B, T=T, N=N)
        return list(map(_reshape_encoding, xs))

    def forward(self, X_sup, Y_sup, X_query):
        if len(X_query.size()) == 4:
            X_query = X_query.unsqueeze(0).unsqueeze(0)
        B, T, N, _, H, W = X_sup.size()
        Bq, Tq, Nq = parse_BTN(X_query)

        # import pdb; pdb.set_trace()
        X_sup = from_6d_to_4d(X_sup)  # (B T N) C H W
        Y_sup = from_6d_to_4d(Y_sup)
        X_query = from_6d_to_4d(X_query)

        X_sup = self.check_image_size(X_sup)
        Y_sup = self.check_image_size(Y_sup)
        X_query = self.check_image_size(X_query)

        # Y_sup = Y_sup - X_sup

        out_X_sup = self.encoder(X_sup)       #out1(BTN C H W), out2(BTN 2C H/2 W/2), out3(BTN 4C H/4 W/4), latent(BTN 8C H/8 W/8)
        out_X_query = self.encoder(X_query)
        out_Y_sup = self.encoder_label(Y_sup)

        # # (B T N) C H W -> B T N C H W
        out_X_sup = self.reshape_encoding(out_X_sup, B=B, T=T, N=N) 
        out_X_query = self.reshape_encoding(out_X_query, B=Bq, T=Tq, N=Nq)
        out_Y_sup = self.reshape_encoding(out_Y_sup, B=B, T=T, N=N)

        matched = self.matching_module(out_X_query, out_X_sup, out_Y_sup)
        matched = from_6d_to_4d(matched)

        out = self.decoder(*matched, residual=from_6d_to_4d(out_X_query) if self.use_residual else None)
        out = from_4d_to_6d(out, B=Bq, T=Tq, N=Nq)
        
        return out[:, :, :H, :W]#, out_deg_Q[:, :, :H, :W]#, out_deg_S[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

