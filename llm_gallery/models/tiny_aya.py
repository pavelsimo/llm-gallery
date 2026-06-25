"""Tiny Aya (3.35B) — Cohere Labs, 2026. A dense model with PARALLEL transformer blocks.

Every other dense model here runs attention and the feed-forward network in series:

    x = x + attn(norm1(x))
    x = x + ffn(norm2(x))

Tiny Aya (like GPT-J and PaLM) runs them **in parallel** from a single shared norm, summing both
into the residual:

    h = norm(x)
    x = x + attn(h) + ffn(h)

This removes one sequential dependency per layer (attn and ffn can be computed at the same time) and
uses one norm per block instead of two. Otherwise it's the standard modern recipe: RoPE, GQA, SwiGLU,
RMSNorm, tied embeddings.

Diagram: https://sebastianraschka.com/llm-architecture-gallery (Tiny Aya)
Tech report: https://arxiv.org/pdf/2603.11510

Self-contained: every module below is defined in THIS file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "Tiny Aya (3.35B)"
RELEASE_DATE = "2026-02-13"
GALLERY_URL = "https://sebastianraschka.com/llm-architecture-gallery"
TECH_REPORT_URL = "https://arxiv.org/pdf/2603.11510"


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 256000
    context_length: int = 8192
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 8
    n_embd: int = 2048
    head_dim: int = 128
    intermediate_size: int = 8192
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True


PRESETS: dict[str, Config] = {
    "tiny": Config(
        vocab_size=65, context_length=128, n_layer=4, n_head=4, n_kv_head=2,
        n_embd=128, head_dim=32, intermediate_size=256, rope_theta=10000.0,
    ),
    "tiny-aya": Config(),
}
DEFAULT_PRESET = "tiny-aya"


# --------------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square normalization; scales tokens without centering."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def precompute_rope(head_dim, max_seq_len, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos, sin = cos[None, None], sin[None, None]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


class Attention(nn.Module):
    """Standard GQA with RoPE; used in parallel with SwiGLU from a single shared norm."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask[:t, :t], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    """Gated FFN: SiLU(gate(x)) ⊙ up(x), projected back to n_embd. No bias."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Parallel block: attention and FFN both read ONE shared norm; their outputs are summed."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask):
        h = self.norm(x)
        return x + self.attn(h, cos, sin, mask) + self.ffn(h)


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class Model(nn.Module):
    """Full language model: embed tokens, run parallel blocks, project to logits."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
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
        b, t = idx.shape
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for block in self.blocks:
            x = block(x, cos, sin, self.causal_mask)
        return self.lm_head(self.norm(x))


# --------------------------------------------------------------------------------------------------
# Standalone smoke test: `python llm_gallery/models/tiny_aya.py`
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(idx)
    F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1)).backward()
    print(f"{MODEL_NAME}  (tiny preset, parallel blocks)")
    print(f"  params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  logits : {tuple(logits.shape)}")
