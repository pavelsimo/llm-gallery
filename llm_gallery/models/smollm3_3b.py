"""SmolLM3 (3B) — Hugging Face, 2025.

SmolLM3 is essentially the Llama recipe (RoPE + GQA + SwiGLU + RMSNorm, tied embeddings) with one
twist worth studying: **NoPE (No Positional Encoding) on a periodic subset of layers**. Every 4th
layer applies *no* RoPE at all — the causal mask alone tells those layers about token order. Mixing
NoPE layers in has been found to improve length generalization (extrapolating past the trained
context) while the remaining RoPE layers still give precise relative positioning.

Diagram: https://sebastianraschka.com/llm-architecture-gallery (SmolLM3)
Tech report: https://huggingface.co/blog/smollm3

Self-contained: every module below is defined in THIS file. It's the closest sibling to
`llama3_8b.py` — diff them to isolate exactly what NoPE changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "SmolLM3 (3B)"
RELEASE_DATE = "2025-06-19"
GALLERY_URL = "https://sebastianraschka.com/llm-architecture-gallery"
TECH_REPORT_URL = "https://huggingface.co/blog/smollm3"


@dataclass
class Config:
    vocab_size: int = 128256
    context_length: int = 65536
    n_layer: int = 36
    n_head: int = 16
    n_kv_head: int = 4
    n_embd: int = 2048
    intermediate_size: int = 11008
    rope_theta: float = 2_000_000.0
    nope_every: int = 4  # every Nth layer (1-indexed) uses NO positional encoding
    norm_eps: float = 1e-5
    tie_embeddings: bool = True


PRESETS: dict[str, Config] = {
    "tiny": Config(
        vocab_size=65, context_length=128, n_layer=4, n_head=4, n_kv_head=2,
        n_embd=128, intermediate_size=256, rope_theta=10000.0, nope_every=4,
    ),
    "smollm3-3b": Config(),
}
DEFAULT_PRESET = "smollm3-3b"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
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
    def __init__(self, cfg: Config, use_rope: bool):
        super().__init__()
        self.use_rope = use_rope  # NoPE layers set this False and skip the rotation
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = apply_rope(q, k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask[:t, :t], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: Config, use_rope: bool):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg, use_rope)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # use_rope is False on every nope_every-th layer (1-indexed) -> those layers are NoPE.
        self.blocks = nn.ModuleList(
            Block(cfg, use_rope=((i + 1) % cfg.nope_every != 0)) for i in range(cfg.n_layer)
        )
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope(head_dim, cfg.context_length, cfg.rope_theta)
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


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(idx)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1))
    loss.backward()
    n_nope = sum(not b.attn.use_rope for b in model.blocks)
    print(f"{MODEL_NAME}  (tiny preset)")
    print(f"  params  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  NoPE    : {n_nope}/{cfg.n_layer} layers have no positional encoding")
    print(f"  logits  : {tuple(logits.shape)}")
    print(f"  loss    : {loss.item():.4f}")
