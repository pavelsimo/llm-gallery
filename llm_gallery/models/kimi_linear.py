"""Kimi Linear (48B-A3B) — Moonshot AI, 2025. A linear-attention + full-attention hybrid.

Most layers use **linear attention**: the softmax over keys is replaced by a feature map ``φ`` (here
``elu(x)+1``, which keeps things positive), turning attention into a running sum that costs O(T) and
O(1) memory instead of O(T²):

    S_t = Σ_{s≤t} φ(k_s) v_sᵀ        z_t = Σ_{s≤t} φ(k_s)        # running key-value memory + normalizer
    o_t = (φ(q_t) S_t) / (φ(q_t) · z_t)

A few layers keep ordinary full attention so the model retains exact long-range lookups; the
feed-forward is a sparse MoE. The recurrence is written sequentially for clarity.

ASSUMPTION: the real Kimi Linear uses MLA for its full-attention layers (see `deepseek_v3.py` for
MLA) and a more elaborate gated "Kimi Delta Attention"; here the full-attention layers are plain GQA
and the linear mixer is vanilla linear attention. The real preset still keeps the published outer
dimensions (27 layers, 2304 hidden size, 256 routed experts, top-8, 1024-wide experts, 1M-token
context, theta=10000); only the internal MLA/KDA mechanics are simplified. The headline idea, O(T)
linear attention interleaved with full attention, is faithful.

Diagram: https://sebastianraschka.com/llm-architecture-gallery (Kimi Linear)
Tech report: https://arxiv.org/pdf/2510.26692

Self-contained: every module below is defined in THIS file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "Kimi Linear (48B-A3B)"
RELEASE_DATE = "2025-10-30"
GALLERY_URL = "https://sebastianraschka.com/llm-architecture-gallery"
TECH_REPORT_URL = "https://arxiv.org/pdf/2510.26692"


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 163840
    context_length: int = 1_000_000
    n_layer: int = 27
    n_embd: int = 2304
    linear_n_head: int = 32
    n_head: int = 32
    n_kv_head: int = 32
    head_dim: int = 72
    rope_theta: float = 10_000.0
    attn_every: int = 4  # every Nth layer (1-indexed) is full attention; rest are linear attention
    n_experts: int = 256
    n_experts_per_tok: int = 8
    moe_intermediate_size: int = 1024
    n_shared_experts: int = 1
    norm_eps: float = 1e-5
    tie_embeddings: bool = False


PRESETS: dict[str, Config] = {
    "tiny": Config(
        vocab_size=65, context_length=128, n_layer=6, n_embd=128, linear_n_head=4, n_head=4,
        n_kv_head=2, head_dim=32, rope_theta=10000.0, attn_every=3, n_experts=8, n_experts_per_tok=2,
        moe_intermediate_size=128, n_shared_experts=1,
    ),
    "kimi-linear": Config(),
}
DEFAULT_PRESET = "kimi-linear"


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


class LinearAttention(nn.Module):
    """O(T) attention via a positive feature map and a running key/value sum (explicit recurrence)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.linear_n_head
        self.head_dim = cfg.n_embd // cfg.linear_n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.g_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.out_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    @staticmethod
    def phi(x):  # positive feature map
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        nh, hd = self.n_head, self.head_dim
        q = self.phi(self.q_proj(x).view(b, t, nh, hd))
        k = self.phi(self.k_proj(x).view(b, t, nh, hd))
        v = self.v_proj(x).view(b, t, nh, hd)

        s = x.new_zeros(b, nh, hd, hd)  # running Σ φ(k) vᵀ
        z = x.new_zeros(b, nh, hd)  # running Σ φ(k)
        outs = []
        for i in range(t):
            k_i, v_i, q_i = k[:, i], v[:, i], q[:, i]  # [B, nh, hd]
            s = s + k_i.unsqueeze(-1) @ v_i.unsqueeze(-2)  # [B, nh, hd_k, hd_v]
            z = z + k_i
            num = (q_i.unsqueeze(-2) @ s).squeeze(-2)  # [B, nh, hd_v]
            den = (q_i * z).sum(-1, keepdim=True).clamp(min=1e-4)
            outs.append(num / den)
        o = torch.stack(outs, dim=1)  # [B, T, nh, hd]
        o = self.out_norm(o).reshape(b, t, d) * F.silu(self.g_proj(x))
        return self.out_proj(o)


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
    """Standard GQA used for the periodic full-attention layers in the linear/full hybrid."""

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


class MLP(nn.Module):
    """SwiGLU expert MLP used as a single routed expert in the MoE layer."""

    def __init__(self, n_embd, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate, bias=False)
        self.up_proj = nn.Linear(n_embd, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """Sparse top-k MoE with optional always-on shared expert."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_experts, self.top_k = cfg.n_experts, cfg.n_experts_per_tok
        self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(MLP(cfg.n_embd, cfg.moe_intermediate_size) for _ in range(cfg.n_experts))
        self.shared = (
            MLP(cfg.n_embd, cfg.moe_intermediate_size * cfg.n_shared_experts)
            if cfg.n_shared_experts > 0 else None
        )

    def forward(self, x):
        b, t, c = x.shape
        x = x.reshape(-1, c)
        topw, topi = F.softmax(self.gate(x), dim=-1).topk(self.top_k, dim=-1)
        topw = topw / topw.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            sel = topi == e
            if not sel.any():
                continue
            ti, slot = sel.nonzero(as_tuple=True)
            out[ti] += topw[ti, slot].unsqueeze(-1) * self.experts[e](x[ti])
        if self.shared is not None:
            out = out + self.shared(x)
        return out.reshape(b, t, c)


class Block(nn.Module):
    """Hybrid block: LinearAttention or GQA (is_attn), always followed by a sparse MoE FFN."""

    def __init__(self, cfg: Config, is_attn: bool):
        super().__init__()
        self.is_attn = is_attn
        self.mix_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mix = Attention(cfg) if is_attn else LinearAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = MoE(cfg)

    def forward(self, x, cos, sin, mask):
        if self.is_attn:
            x = x + self.mix(self.mix_norm(x), cos, sin, mask)
        else:
            x = x + self.mix(self.mix_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class Model(nn.Module):
    """Full language model: embed tokens, run linear/full-attention hybrid blocks, project to logits."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            Block(cfg, is_attn=((i + 1) % cfg.attn_every == 0)) for i in range(cfg.n_layer)
        )
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
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for block in self.blocks:
            x = block(x, cos, sin, self.causal_mask)
        return self.lm_head(self.norm(x))


# --------------------------------------------------------------------------------------------------
# Standalone smoke test: `python llm_gallery/models/kimi_linear.py`
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(idx)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1))
    loss.backward()
    n_attn = sum(b.is_attn for b in model.blocks)
    print(f"{MODEL_NAME}  (tiny preset)")
    print(f"  mixers : {cfg.n_layer - n_attn} linear-attention + {n_attn} full-attention layers")
    print(f"  params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  logits : {tuple(logits.shape)}  loss {loss.item():.4f}  (~ln(vocab)={math.log(cfg.vocab_size):.4f})")
