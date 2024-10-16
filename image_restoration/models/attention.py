import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

from .reshape import from_6d_to_4d, from_4d_to_6d, get_reshaper


class TrAttentionBase(nn.Module):  # spatial
    def __init__(self, dim: int, num_heads: int, bias: bool = True, residual: bool = False):
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
            self.aggregate_residual = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

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

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
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
    def __init__(self, dim: int, num_heads: int, bias: bool = True, k: float = -1, residual: bool = False):
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
    def __init__(self, dim: int, num_heads: int, last_dim: int, bias: bool = True, k: float = -1,
                 residual: bool = False):
        super().__init__(dim, num_heads, bias, k, residual=residual)
        self.norm1_q = nn.LayerNorm(last_dim)
        self.norm1_k = nn.LayerNorm(last_dim)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
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


class TopkLearningModule(nn.Module):
    def __init__(self, num_features, hidden_features=128):
        super().__init__()
        self.params = nn.Parameter(torch.rand(hidden_features))

        self.net1 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        self.net2 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, hidden_features)
        )
        self.head = nn.Linear(hidden_features, num_features)
        self.relu = nn.ReLU()

    def forward(self):
        x = self.net1(self.params)
        x = self.relu(x + self.params)
        x = self.relu(self.net2(x) + x)
        x = self.head(x)
        return x.clamp(min=0, max=1)  # top-k mask


class TrLearnableTopkAttention(TrAttentionBase):
    def __init__(self, dim: int, num_heads: int, k_dim: int, bias: bool = True, residual: bool = False):
        super().__init__(dim, num_heads, bias, residual=residual)
        self.topk_net = TopkLearningModule(k_dim)

    def topk(self, attn: torch.Tensor) -> torch.Tensor:
        vals, idxs = attn.sort(dim=-1, descending=False)  # B H NqX NsX
        vals = vals - vals[..., 0, None]  # now all >= 0
        mask = self.topk_net()
        vals = vals * mask
        vals[vals <= 0] = float('-inf')
        attn = attn.fill_(float('-inf')).scatter_(-1, idxs, vals)
        return attn

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = self.topk(attn)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out


class TrLearnableTopkLayerNormAttention(TrLearnableTopkAttention):
    def __init__(self, dim: int, num_heads: int, k_dim: int, last_dim: int, bias: bool = True, residual: bool = False):
        super().__init__(dim, num_heads, k_dim, bias, residual=residual)
        self.norm1_q = nn.LayerNorm(last_dim)
        self.norm1_k = nn.LayerNorm(last_dim)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
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


class TrLayerNormAttention(TrAttentionBase):
    def __init__(self, dim: int, num_heads: int, last_dim: int, bias: bool = True, residual: bool = False):
        super().__init__(dim, num_heads, bias, residual=residual)
        self.norm1_q = nn.LayerNorm(last_dim)
        self.norm1_k = nn.LayerNorm(last_dim)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
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


class TrSpAttention_FixedTopK(TrFixedTopkLayerNormAttention):  # spatial
    def __init__(self, dim: int, num_heads: int, last_dim: int, bias=True, k=-1, residual: bool = False):
        super(TrSpAttention_FixedTopK, self).__init__(dim, num_heads, last_dim, bias, k, residual=residual)

        self.transpose = get_reshaper('(B T N) (head C) H W -> (B T) head (N H W) C')
        self.transpose_out = get_reshaper('(B T) head (N H W) C -> (B T N) (head C) H W')


class TrSpAttention_LearnableTopK(TrLearnableTopkLayerNormAttention):  # spatial
    def __init__(self, dim: int, num_heads: int, k_dim: int, last_dim: int, bias=True, residual: bool = False):
        super(TrSpAttention_LearnableTopK, self).__init__(dim, num_heads, k_dim, last_dim, bias, residual=residual)

        self.transpose = get_reshaper('(B T N) (head C) H W -> (B T) head (N H W) C')
        self.transpose_out = get_reshaper('(B T) head (N H W) C -> (B T N) (head C) H W')


class TrChanAttention_FixedTopK(TrFixedTopkLayerNormAttention):  # channel
    def __init__(self, dim: int, num_heads: int, last_dim: int, bias=True, k=-1, residual: bool = False):
        super(TrChanAttention_FixedTopK, self).__init__(dim, num_heads, last_dim, bias, k, residual=residual)

        self.transpose = get_reshaper('(B T N) (head C) H W -> (B T) head (N C) (H W)')
        self.transpose_out = get_reshaper('(B T) head (N C) (H W) -> (B T N) (head C) H W')


class TrChanAttention_LearnableTopK(TrLearnableTopkLayerNormAttention):  # channel
    def __init__(self, dim: int, num_heads: int, k_dim: int, last_dim: int, bias=True, residual: bool = False):
        super(TrChanAttention_LearnableTopK, self).__init__(dim, num_heads, k_dim, last_dim, bias, residual=residual)

        self.transpose = get_reshaper('(B T N) (head C) H W -> (B T) head (N C) (H W)')
        self.transpose_out = get_reshaper('(B T) head (N C) (H W) -> (B T N) (head C) H W')


class TrChanAttention_Base(TrLayerNormAttention):  # channel
    def __init__(self, dim: int, num_heads: int, last_dim: int, bias=True, residual: bool = False):
        super(TrChanAttention_Base, self).__init__(dim, num_heads, last_dim, bias, residual=residual)

        self.transpose = get_reshaper('(B T N) (head C) H W -> (B T) head (N C) (H W)')
        self.transpose_out = get_reshaper('(B T) head (N C) (H W) -> (B T N) (head C) H W')


# Deprecated
class TrReducedSpAttention(TrSpAttention_FixedTopK):  # spatial
    def __init__(self, dim, num_heads, bias=True, k=-1, r=2):
        raise Exception("This class is deprecated. Please check before use")
        super(TrReducedSpAttention, self).__init__(dim, num_heads, bias, k)
        self.conv2_q = nn.Conv2d(dim, dim * 2 * r, kernel_size=2, stride=r, bias=bias)
        self.conv2_k = nn.Conv2d(dim, dim * 2 * r, kernel_size=2, stride=r, bias=bias)
        self.conv2_v = nn.Conv2d(dim, dim * 2 * r, kernel_size=2, stride=r, bias=bias)

        self.upscale = nn.PixelShuffle(2)
        self.project_out = nn.Conv2d(dim * 2 * r, dim * 2 * r, kernel_size=1, bias=bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        Bq, Tq, Nq = q.shape[:3]

        q, k, v, shape = self.project_input(q, k, v)

        q = self.conv2_q(q)
        k = self.conv2_k(k)
        v = self.conv2_v(v)
        out = self.attention(q, k, v, shape)

        out = self.project_out(out)
        out = self.upscale(out)
        out = from_4d_to_6d(out, B=Bq, T=Tq, N=Nq)
        return out
