"""MiniMax M3 (428B) — MiniMax AI, 2026.

MiniMax M3 extends the M2-style sparse MoE backbone to million-token context. The public config has:

  * 3 dense prefix layers followed by 57 sparse MoE layers.
  * GQA with QK-Norm (64 query heads, 4 KV heads).
  * MiniMax Sparse Attention metadata after the prefix layers.
  * 128 routed experts, top-4 routing, plus one shared expert.

ASSUMPTION: the sparse block-index selection is represented in config/docstrings and the visualizer,
but this runnable educational implementation uses full causal attention for the attention math.

Diagram: https://sebastianraschka.com/llm-architecture-gallery (MiniMax M3)
Tech report: https://arxiv.org/abs/2606.13392
Config: https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json

Self-contained: every module below is defined in THIS file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "MiniMax M3 (428B)"
RELEASE_DATE = "2026-06-13"
GALLERY_URL = "https://sebastianraschka.com/llm-architecture-gallery"
TECH_REPORT_URL = "https://arxiv.org/abs/2606.13392"


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 200064
    context_length: int = 1048576
    n_layer: int = 60
    first_k_dense: int = 3
    n_head: int = 64
    n_kv_head: int = 4
    n_embd: int = 6144
    head_dim: int = 128
    dense_intermediate_size: int = 12288
    moe_intermediate_size: int = 3072
    n_experts: int = 128
    n_experts_per_tok: int = 4
    n_shared_experts: int = 1
    sparse_attention_topk_blocks: int = 16
    sparse_attention_block_size: int = 128
    rope_theta: float = 5000000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = False


PRESETS: dict[str, Config] = {
    "tiny": Config(
        vocab_size=65,
        context_length=128,
        n_layer=4,
        first_k_dense=1,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        head_dim=32,
        dense_intermediate_size=256,
        moe_intermediate_size=128,
        n_experts=8,
        n_experts_per_tok=2,
        n_shared_experts=1,
        sparse_attention_topk_blocks=4,
        sparse_attention_block_size=16,
        rope_theta=10000.0,
    ),
    "minimax-m3": Config(),
}
DEFAULT_PRESET = "minimax-m3"


# --------------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square normalization; scales tokens without centering."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos, sin = cos[None, None], sin[None, None]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


class MiniMaxSparseAttention(nn.Module):
    """GQA + QK-Norm; sparse index selection is approximated by full causal attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.sparse_topk_blocks = cfg.sparse_attention_topk_blocks
        self.sparse_block_size = cfg.sparse_attention_block_size
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.n_embd, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask[:t, :t], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    """Gated MLP used by dense prefix layers and routed/shared experts."""

    def __init__(self, n_embd: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate, bias=False)
        self.up_proj = nn.Linear(n_embd, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """Sigmoid-routed MoE with a shared always-on expert."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.top_k = cfg.n_experts_per_tok
        self.n_experts = cfg.n_experts
        self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(cfg.n_embd, cfg.moe_intermediate_size) for _ in range(cfg.n_experts)
        )
        self.shared = (
            SwiGLU(cfg.n_embd, cfg.moe_intermediate_size * cfg.n_shared_experts)
            if cfg.n_shared_experts > 0
            else None
        )

    def forward(self, x):
        b, t, c = x.shape
        x = x.reshape(-1, c)
        scores = self.gate(x).sigmoid()
        topw, topi = scores.topk(self.top_k, dim=-1)
        topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-20)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            sel = topi == e
            if not sel.any():
                continue
            tok_idx, slot = sel.nonzero(as_tuple=True)
            out[tok_idx] += topw[tok_idx, slot].unsqueeze(-1) * self.experts[e](x[tok_idx])
        if self.shared is not None:
            out = out + self.shared(x)
        return out.reshape(b, t, c)


class Block(nn.Module):
    """Pre-norm block: QK-Norm GQA + dense prefix MLP or sparse MoE FFN."""

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = MiniMaxSparseAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        if layer_idx < cfg.first_k_dense:
            self.ffn = SwiGLU(cfg.n_embd, cfg.dense_intermediate_size)
        else:
            self.ffn = MoE(cfg)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class Model(nn.Module):
    """Full language model: embed tokens, run dense-prefix/MoE blocks, project to logits."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = precompute_rope(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        causal = torch.triu(
            torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        assert t <= self.config.context_length, f"sequence length {t} > context {self.config.context_length}"
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for block in self.blocks:
            x = block(x, cos, sin, self.causal_mask)
        return self.lm_head(self.norm(x))


# --------------------------------------------------------------------------------------------------
# Standalone smoke test: `python llm_gallery/models/minimax_m3.py`
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(idx)
    F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1)).backward()
    print(f"{MODEL_NAME}  (tiny preset)")
    print(f"  params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  logits : {tuple(logits.shape)}")
