import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    n_layer: int = 4
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.1
    bias: bool = False
    is_causal: bool = False
    src_dim: int = 7
    src_len: int = 40
    out_dim: int = 2
    num_predictions: int = 1
    device: Optional[str] = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class TransformerModel(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.model_type = 'TransformerModel'
        self.device = torch.device(cfg.device)

        self.transformer = nn.ModuleDict(dict(
            src_enc=nn.Linear(cfg.src_dim, cfg.n_embd),
            time_emb=AbsolutePositionalEmbedding(dim=cfg.n_embd, max_seq_len=cfg.src_len),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, bias=cfg.bias),
        )).to(self.device)

        self.K = cfg.num_predictions
        self.dec = nn.Linear(cfg.n_embd, self.K * cfg.out_dim).to(self.device)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.transformer.src_enc(src.to(self.device))  # Embed
        src = src + self.transformer.time_emb(src)  # Add time embedding (positional encoding)

        for block in self.transformer.h:
            src = block(src)

        # Predicted input trajectory discard the last timestep
        out = self.dec(src)  # [B, src_len, K * in_trj_dim]
        out = out.unflatten(-1, (self.K, self.cfg.out_dim))  # [B, src_len, K, out_dim]
        return out


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention layer for Transformer models.
    Adapted from: https://github.com/karpathy/nanoGPT/blob/master/model.py

    Dimension key:
    B: batch size
    L: sequence length
    D: model dimension (sometimes called d_model or embedding_dim)
    H: number of attention heads in a layer

    The name of a tensor ends in a dimension-suffix composed of these letters, e.g. x_BLD for a three-dimensional tensor with batch, length, and model dimension.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # output projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        # regularization
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.is_causal = cfg.is_causal

    def forward(self, x_BLD: torch.Tensor):
        B, L, D = x_BLD.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q_BLD, k_BLD, v_BLD = self.c_attn(x_BLD).split(self.n_embd, dim=2)
        k_BNHD = k_BLD.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q_BNHD = q_BLD.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v_BNHD = v_BLD.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal self-attention; Self-attend: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        # efficient attention using Flash Attention CUDA kernels
        y_BNLT = F.scaled_dot_product_attention(q_BNHD, k_BNHD, v_BNHD, attn_mask=None,
                                                dropout_p=self.dropout if self.training else 0,
                                                is_causal=self.is_causal)

        y_BLD = y_BNLT.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        y_BLD = self.resid_dropout(self.c_proj(y_BLD))
        return y_BLD


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) layer for Transformer models.
    Adapted from: https://github.com/karpathy/nanoGPT/blob/master/model.py

    Dimension key:
    B: batch size
    L: sequence length
    D: model dimension (sometimes called d_model or embedding_dim)

    The name of a tensor ends in a dimension-suffix composed of these letters, e.g. x_BLD for a three-dimensional tensor with batch, length, and model dimension.
    """

    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x_BLD: torch.Tensor):
        x_BLD = self.c_fc(x_BLD)
        x_BLD = self.gelu(x_BLD)
        x_BLD = self.c_proj(x_BLD)
        x_BLD = self.dropout(x_BLD)
        return x_BLD


class Block(nn.Module):
    """
    Transformer block layer.
    Adapted from: https://github.com/karpathy/nanoGPT/blob/master/model.py

    Dimension key:
    B: batch size
    L: sequence length
    D: model dimension (sometimes called d_model or embedding_dim)

    The name of a tensor ends in a dimension-suffix composed of these letters, e.g. x_BLD for a three-dimensional tensor with batch, length, and model dimension.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x_BLD: torch.Tensor):
        x_BLD = x_BLD + self.attn(self.ln_1(x_BLD))
        x_BLD = x_BLD + self.mlp(self.ln_2(x_BLD))
        return x_BLD


class LayerNorm(nn.Module):
    """
    Layer normalization layer but with an optional bias. PyTorch doesn't support simply bias=False
    Adapted from: https://github.com/karpathy/nanoGPT/blob/master/model.py

    Dimension key:
    B: batch size
    L: sequence length
    D: model dimension (sometimes called d_model or embedding_dim)

    The name of a tensor ends in a dimension-suffix composed of these letters, e.g. x_BLD for a three-dimensional tensor with batch, length, and model dimension.
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_BLD: torch.Tensor):
        return F.layer_norm(input_BLD, self.weight.shape, self.weight, self.bias, 1e-5)


class AbsolutePositionalEmbedding(nn.Module):
    """
    Absolute positional embedding layer.
    Adapted from: x-transformers
    """

    def __init__(self, dim, max_seq_len, l2norm_embed=False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, (f'you are passing in a sequence length of {seq_len}'
                                             f' but your absolute positional embedding has a max sequence length of'
                                             f' {self.max_seq_len}')

        pos = torch.arange(seq_len, device=device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return self.l2norm(pos_emb) if self.l2norm_embed else pos_emb

    @staticmethod
    def l2norm(t):
        return F.normalize(t, p=2, dim=-1)
