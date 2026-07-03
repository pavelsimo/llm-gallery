"""OLMo 2 (7B) — Allen AI, 2024.

A fully-open dense model whose two notable departures from the Llama recipe are worth studying:

  1. **QK-Norm**: apply RMSNorm to the *whole* query and key projections (before splitting into
     heads). This bounds the magnitude of attention logits and noticeably stabilizes training.
  2. **Post-norm placement (\"reordered norm\")**: instead of normalizing the *input* of each sublayer
     (pre-norm, as in Llama), OLMo 2 normalizes the *output* of the sublayer before the residual add:

         x = x + norm(attn(x))      # attn sees the raw residual stream; norm is applied after
         x = x + norm(ffn(x))

OLMo 2 7B keeps full multi-head attention (no GQA), RoPE, and a SwiGLU MLP.
The same body also supports OLMo 3's 3-sliding/1-full attention schedule through config fields;
OLMo 2 leaves that schedule disabled.

Diagram: https://sebastianraschka.com/llm-architecture-gallery (OLMo 2)
Tech report: https://arxiv.org/pdf/2501.00656

Self-contained: every module below is defined in THIS file. Compare it side-by-side with
`llama3_8b.py` to see exactly what QK-Norm and post-norm change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "OLMo 2 (7B)"
RELEASE_DATE = "2024-11-25"
GALLERY_URL = "https://sebastianraschka.com/llm-architecture-gallery"
TECH_REPORT_URL = "https://arxiv.org/pdf/2501.00656"


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 100352
    context_length: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 32  # == n_head -> full MHA (OLMo 2 7B does not use GQA)
    n_embd: int = 4096
    intermediate_size: int = 11008
    rope_theta: float = 500000.0
    sliding_window: int = 0  # 0 = full causal attention in every layer
    global_every: int = 1  # with sliding_window > 0, every Nth layer is full/global
    norm_eps: float = 1e-6


PRESETS: dict[str, Config] = {
    "tiny": Config(
        vocab_size=65, context_length=128, n_layer=4, n_head=4, n_kv_head=4,
        n_embd=128, intermediate_size=256, rope_theta=10000.0,
    ),
    "olmo2-7b": Config(),
}
DEFAULT_PRESET = "olmo2-7b"


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
        return (x.to(dtype)) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
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


class Attention(nn.Module):
    """Full MHA (no GQA) with OLMo-style QK-Norm over the entire projection, not per-head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)
        # QK-Norm over the FULL projection dimension (OLMo 2 style), not per-head.
        self.q_norm = RMSNorm(self.n_head * self.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(self.n_kv_head * self.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x))  # normalize the whole [B, T, n_head*hd] vector
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)

        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

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
    """Post-norm block: the norm is applied to the sublayer *output*, not its input."""

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.is_global = cfg.sliding_window <= 0 or (layer_idx + 1) % cfg.global_every == 0
        self.attn = Attention(cfg)
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn_norm(self.attn(x, cos, sin, mask))
        x = x + self.ffn_norm(self.ffn(x))
        return x


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class Model(nn.Module):
    """Full language model: embed tokens, run post-norm blocks, project to logits."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope(head_dim, cfg.context_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        n = cfg.context_length
        ar = torch.arange(n)
        future = ar[None, :] > ar[:, None]
        if cfg.sliding_window > 0:
            too_far = (ar[:, None] - ar[None, :]) >= cfg.sliding_window
            self.register_buffer("mask_local", future | too_far, persistent=False)
        else:
            self.register_buffer("mask_local", future, persistent=False)
        self.register_buffer("mask_global", future, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for block in self.blocks:
            mask = self.mask_global if block.is_global else self.mask_local
            x = block(x, cos, sin, mask)
        return self.lm_head(self.norm(x))


# --------------------------------------------------------------------------------------------------
# Standalone smoke test: `python llm_gallery/models/olmo2_7b.py`
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(idx)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1))
    loss.backward()
    print(f"{MODEL_NAME}  (tiny preset)")
    print(f"  params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  logits : {tuple(logits.shape)}")
    print(f"  loss   : {loss.item():.4f}")
