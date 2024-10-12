import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

from .reshape import from_6d_to_4d, from_4d_to_6d, get_reshaper


class TrAttentionBase(nn.Module): #spatial
    def __init__(self, dim: int, num_heads: int, bias: bool=True, residual: bool=False):
        super(TrAttentionBase, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv1_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv1_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv1_v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.dwconv_q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv_k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.residual = residual
        if residual:
            self.aggregate_residual = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)

        self.transpose = get_reshaper('(B T N) (head C) H W -> (B T) head (N C) (H W)')
        self.transpose_out = get_reshaper('(B T) head (N C) (H W) -> (B T N) (head C) H W')

    def project_input(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = self.dwconv_q(self.conv1_q(q))
        k = self.dwconv_k(self.conv1_k(k))
        v = self.dwconv_v(self.conv1_v(v))

        return q, k, v

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1) 
        out = attn @ v
        return out

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        B, _, _, Nq, N = shape

        q = self.transpose(q, head=self.num_heads, B=B, N=Nq)
        k = self.transpose(k, head=self.num_heads, B=B, N=N)
        v = self.transpose(v, head=self.num_heads, B=B, N=N)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        out = self._attention(q, k, v)
        out = self.transpose_out(out, head=self.num_heads, B=B, N=Nq)
        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        Bq, Tq, Nq = q.shape[:3]
        B, T, N = k.shape[:3]

        q = from_6d_to_4d(q)
        k = from_6d_to_4d(k)
        v = from_6d_to_4d(v)

        qi, ki, vi = self.project_input(q, k, v)
        out = self.attention(qi, ki, vi, (B, Tq, T, Nq, N))

        out = self.project_out(out)

        if self.residual:
            out = torch.cat((out, q), dim=1)
            out = self.aggregate_residual(out)

        out = from_4d_to_6d(out, B=Bq, T=Tq, N=Nq)
        return out


class TrFixedTopkAttention(TrAttentionBase):
    def __init__(self, dim: int, num_heads: int, bias: bool=True, k: float = -1, residual: bool=False):
        super().__init__(dim, num_heads, bias, residual=residual)
        self.topk_value = k
    
    def topk(self, attn: torch.Tensor) -> torch.Tensor:
        """
        topk for each image
        * attn : Bq head NqXq NsXs
        """
        if self.topk_value < 0:
            return attn

        k = int(attn.shape[-1] * self.topk_value) if self.topk_value < 1 else int(self.topk_value)

        vals, idxs = attn.topk(k=k, dim=-1)
        attn = attn.fill_(float('-inf')).scatter_(-1, idxs, vals)
        return attn

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.topk(attn)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out

class TrFixedTopkLayerNormAttention(TrFixedTopkAttention):
    def __init__(self, dim: int, num_heads: int, last_dim: int, bias: bool=True, k: float = -1, residual: bool=False):
        super().__init__(dim, num_heads, bias, k, residual=residual)
        self.norm1_q = nn.LayerNorm(last_dim)
        self.norm1_k = nn.LayerNorm(last_dim)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        B, _, _, Nq, N = shape
        H, W = q.shape[-2:]

        q = self.transpose(q, head=self.num_heads, B=B, N=Nq)
        k = self.transpose(k, head=self.num_heads, B=B, N=N)
        v = self.transpose(v, head=self.num_heads, B=B, N=N)
        
        nq = self.norm1_q(q)
        nk = self.norm1_k(k)

        out = self._attention(nq, nk, v)
        out = out + q

        out = self.transpose_out(out, head=self.num_heads, B=B, N=Nq, H=H, W=W)
        return out