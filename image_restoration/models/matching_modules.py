import torch
import torch.nn as nn
from einops import rearrange
from typing import List
from .attention import TrSpAttention_LearnableTopK, TrSpAttention_FixedTopK, TrChanAttention_FixedTopK, TrChanAttention_Base, TrReducedSpAttention, TrChanAttention_LearnableTopK


class SpChMatchingModule(nn.Module):
    def __init__(self, dim: List[int], num_heads: List[int], 
                 last_dim_sp: List[int], last_dim_ch: List[int], 
                 num_shots: int,
                 mode: str = 'topk-fixed', topk: List[float]=None, 
                 use_residual: bool = False,
                 reduce_spatial: int=None,):
        super().__init__()

        self.n_levels = len(dim) # 4
        if topk is None:
            topk = [-1] * self.n_levels

        assert mode in ['topk-fixed', 'topk-fixed-sponly', 'topk-learn', 'topk-learn-sponly', 'none']
        m = mode.split('-')
        if m[0] == 'topk':
            if m[1] == 'fixed':
                if len(m) > 2 and m[2] == 'sponly':
                    self.matching_chan = nn.ModuleList([TrChanAttention_Base(d, num_heads=n, last_dim=l, residual=use_residual) for d, n, l in zip(dim, num_heads, last_dim_ch)])
                else:
                    self.matching_chan = nn.ModuleList([TrChanAttention_FixedTopK(d, last_dim=l, num_heads=n, k=k, residual=use_residual) for d, l, n, k in zip(dim, last_dim_ch, num_heads, topk)])

                if reduce_spatial is not None:
                    self.matching_sp = nn.ModuleList([TrReducedSpAttention(d, num_heads=n, k=k, r=reduce_spatial) for d, n, k in zip(dim, num_heads, topk)])
                else:
                    self.matching_sp = nn.ModuleList([TrSpAttention_FixedTopK(d, last_dim=d//n, num_heads=n, k=k, residual=use_residual) for d, n, k in zip(dim, num_heads, topk)])

            elif m[1] == 'learn':                
                if len(m) > 2 and m[2] == 'sponly':
                    self.matching_chan = nn.ModuleList([TrChanAttention_Base(d, num_heads=n, last_dim=l, residual=use_residual) for d, n, l in zip(dim, num_heads, last_dim_ch)])
                else:
                    self.matching_chan = nn.ModuleList([TrChanAttention_LearnableTopK(d, num_heads=n, k_dim=k*num_shots//n, last_dim=l, residual=use_residual) for d, n, l, k in zip(dim, num_heads, last_dim_ch, last_dim_sp)])

                self.matching_sp = nn.ModuleList([TrSpAttention_LearnableTopK(d, num_heads=n, k_dim=k*num_shots, last_dim=d//n, residual=use_residual) for d, n, l, k in zip(dim, num_heads, last_dim_sp, last_dim_ch)])
                
        else:
            self.matching_chan = nn.ModuleList([TrChanAttention_Base(d, num_heads=n, last_dim=l, residual=use_residual) for d, n, l in zip(dim, num_heads, last_dim_ch)])
            self.matching_sp = nn.ModuleList([TrSpAttention_LearnableTopK(d, num_heads=n, k_dim=k*num_shots, last_dim=d//n, residual=use_residual) for d, n, l, k in zip(dim, num_heads, last_dim_sp, last_dim_ch)])


    def forward(self, W_query: List[torch.Tensor], W_sup: List[torch.Tensor], Z_sup: List[torch.Tensor]) -> List[torch.Tensor]:  #List[B T N kC H/k W/k]
        assert len(W_query) == self.n_levels, f'Expected level {self.n_levels} but len(Wq) = {len(W_query)}'
            
        Z_Qs = []
        for spat, chan, Q, K, V in zip(self.matching_sp, self.matching_chan, W_query, W_sup, Z_sup):
            Z_Q = spat(Q, K, V)
            Z_Q += chan(Q, K, V)
            Z_Qs.append(Z_Q)
        
        # for chan, Q, K, V in zip(self.matching_chan, W_query, W_sup, Z_sup):
        #     Z_Q = chan(Q, K, V)
        #     Z_Qs.append(Z_Q)
        
        return Z_Qs
    