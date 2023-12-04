## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn

from .Restormer import OverlapPatchEmbed, TransformerBlock, Downsample, Upsample, grad_reverse
from einops import rearrange
from .reshape import from_4d_to_6d, from_6d_to_4d, parse_BTN
from typing import List

# from model.matching_modules import ChannelMatchingModule
class Restormer_Encoder(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias'   ## Other option 'BiasFree'
    ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img):
        # import pdb; pdb.set_trace()
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 

        return out_enc_level1, out_enc_level2, out_enc_level3, latent

##---------- Restormer -----------------------
class Restormer_Decoder(nn.Module):
    def __init__(self, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias'   ## Other option 'BiasFree'
    ):
        super(Restormer_Decoder, self).__init__()

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, out_enc_level1, out_enc_level2, out_enc_level3, latent):
        # import pdb; pdb.set_trace()
        out = self.up4_3(latent)
        out = torch.cat([out, out_enc_level3], 1)
        out = self.reduce_chan_level3(out)
        out = self.decoder_level3(out) 

        out = self.up3_2(out)
        out = torch.cat([out, out_enc_level2], 1)
        out = self.reduce_chan_level2(out)
        out = self.decoder_level2(out) 

        out = self.up2_1(out)
        out = torch.cat([out, out_enc_level1], 1)
        out = self.decoder_level1(out)
        
        out = self.refinement(out)
        out = self.output(out)

        return out

    
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias'   ## Other option 'BiasFree'
    ):

        super(Restormer, self).__init__()
        self.encoder = Restormer_Encoder(inp_channels=inp_channels, dim=dim, num_blocks=num_blocks, heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.decoder = Restormer_Decoder(out_channels=out_channels, dim=dim, num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks, heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, inp_img):
        out_enc_level1, out_enc_level2, out_enc_level3, latent = self.encoder(inp_img)
        out_dec_level1 = self.decoder(out_enc_level1, out_enc_level2, out_enc_level3, latent)
        out_dec_level1 = out_dec_level1 + inp_img

        return out_dec_level1


class Restormer_Fewshot(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 12,
        num_blocks = [1,1,1,1], 
        num_refinement_blocks = 4,
        heads = [1,1,1,1],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias'   ## Other option 'BiasFree'
    ):

        super(Restormer_Fewshot, self).__init__()

        #image encoder (query & support img)
        self.encoder = Restormer_Encoder(inp_channels=inp_channels, dim=dim, num_blocks=num_blocks, heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        #label encoder (support label)
        # self.encoder_label = Restormer_Encoder(inp_channels=inp_channels, dim=dim, num_blocks=num_blocks, heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        self.decoder = Restormer_Decoder(out_channels=out_channels, dim=dim, num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks, heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.matching_module = ChannelMatchingModule(dim=dim, num_heads=heads, n_levels = len(num_blocks))

    def reshape_encoding(self, xs: List[torch.Tensor], B, T, N):
        def _reshape_encoding(x):
            return from_4d_to_6d(x, B=B, T=T, N=N)
        return list(map(_reshape_encoding, xs))

    def forward(self, X_sup, Y_sup, X_query):
        # import pdb; pdb.set_trace()
        B, T, N = parse_BTN(X_sup)
        Bq, Tq, Nq = parse_BTN(X_query)

        X_sup = from_6d_to_4d(X_sup)  # B T N C H W -> (B T N) C H W
        Y_sup = from_6d_to_4d(Y_sup)
        X_query = from_6d_to_4d(X_query)

        out_X_sup = self.encoder(X_sup)         #out1(BTN C H W), out2(BTN 2C H/2 W/2), out3(BTN 4C H/4 W/4), latent(BTN 8C H/8 W/8)
        out_X_query = self.encoder(X_query)
        out_Y_sup = self.encoder(Y_sup)

        # # (B T N) C H W -> B T N C H W
        out_X_sup = self.reshape_encoding(out_X_sup, B=B, T=T, N=N) 
        out_X_query = self.reshape_encoding(out_X_query, B=Bq, T=Tq, N=Nq)
        out_Y_sup = self.reshape_encoding(out_Y_sup, B=B, T=T, N=N)

        matched = self.matching_module(out_X_query, out_X_sup, out_Y_sup)

        matched = from_6d_to_4d(matched)

        # matching here
        out_dec_level1 = self.decoder(*matched) # (B T N) C H W

        out = from_4d_to_6d(out_dec_level1, B=B, T=T, N=N)
        # import pdb; pdb.set_trace()
        out = out + from_4d_to_6d(X_query, B=B, T=T, N=N)

        return out