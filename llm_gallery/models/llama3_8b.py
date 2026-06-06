"""Llama 3 (8B) — Meta, 2024.

The canonical *modern* dense Transformer. Read `gpt2_xl.py` first; Llama 3 is GPT-2 with four parts
swapped out, and almost every other dense model in the gallery is a variation on this recipe:

    GPT-2                         Llama 3
    --------------------------    ----------------------------------------------------
    learned absolute positions -> RoPE (rotary), applied to Q and K inside attention
    LayerNorm (with bias)      -> RMSNorm (no mean-subtraction, no bias)
    full multi-head attention  -> grouped-query attention (GQA): query heads share K/V heads
    dense GELU MLP             -> SwiGLU gated MLP (SiLU(gate) * up -> down)
    biases everywhere          -> no biases in any Linear

Diagram: https://sebastianraschka.com/llm-architecture-gallery (Llama 3)
Tech report: https://arxiv.org/pdf/2407.21783

Self-contained: every module below is defined in THIS file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "Llama 3 (8B)"
RELEASE_DATE = "2024-04-18"
GALLERY_URL = "https://sebastianraschka.com/llm-architecture-gallery"
TECH_REPORT_URL = "https://arxiv.org/pdf/2407.21783"


@dataclass
class Config:
    vocab_size: int = 128256
    context_length: int = 8192
    n_layer: int = 32
    n_head: int = 32  # query heads
    n_kv_head: int = 8  # key/value heads (GQA): each is shared by n_head/n_kv_head query heads
    n_embd: int = 4096
    intermediate_size: int = 14336  # SwiGLU hidden size
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False  # Llama 3 8B uses a separate output projection


PRESETS: dict[str, Config] = {
    "tiny": Config(
        vocab_size=65, context_length=128, n_layer=4, n_head=4, n_kv_head=2,
        n_embd=128, intermediate_size=256, rope_theta=10000.0,
    ),
    "llama3-8b": Config(),
}
DEFAULT_PRESET = "llama3-8b"


# --------------------------------------------------------------------------------------------------
# Normalization: RMSNorm
# --------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm: scale each vector by its RMS, then a learned per-channel gain.

    No mean subtraction and no bias (vs LayerNorm). The reduction is done in float32 for numerical
    stability and cast back, which is what production implementations do.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


# --------------------------------------------------------------------------------------------------
# Positional encoding: RoPE (rotary position embeddings)
# --------------------------------------------------------------------------------------------------
def precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    """Return (cos, sin) tables of shape [max_seq_len, head_dim].

    Each pair of channels is rotated by an angle = position * frequency. Low channels rotate slowly
    (long wavelength), high channels quickly — encoding *relative* position when Q and K interact.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [hd/2]
    t = torch.arange(max_seq_len).float()  # [T]
    freqs = torch.outer(t, inv_freq)  # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, hd] (duplicate so each half lines up)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map [x1, x2] -> [-x2, x1] over the last dim (the two halves of each head vector)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate q and k ([B, n_head, T, hd]) using cos/sin tables ([T, hd])."""
    cos = cos[None, None, :, :]  # [1, 1, T, hd] for broadcasting over batch + heads
    sin = sin[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# --------------------------------------------------------------------------------------------------
# Attention: grouped-query (GQA)
# --------------------------------------------------------------------------------------------------
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand [B, n_kv, T, hd] -> [B, n_kv * n_rep, T, hd] so each K/V head feeds n_rep query heads."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be a multiple of n_kv_head"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head

        # Q is full width; K and V are narrower (only n_kv_head heads) — this is what saves memory.
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)  # [B, nkv, T, hd]
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)  # [B, nh, T, hd]
        v = repeat_kv(v, self.n_rep)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nh, T, T]
        att = att.masked_fill(mask[:t, :t], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # [B, nh, T, hd]

        y = y.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)
        return self.o_proj(y)


# --------------------------------------------------------------------------------------------------
# Feed-forward: SwiGLU
# --------------------------------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """Gated MLP: down( SiLU(gate(x)) * up(x) ). Two input projections instead of GPT-2's one."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE tables and the causal mask once (not saved in checkpoints).
        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope(head_dim, cfg.context_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        causal = torch.triu(
            torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool), diagonal=1
        )  # True above the diagonal = "future", which we mask out
        self.register_buffer("causal_mask", causal, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        assert t <= self.config.context_length
        x = self.tok_emb(idx)  # [B, T, C]
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for block in self.blocks:
            x = block(x, cos, sin, self.causal_mask)
        x = self.norm(x)
        return self.lm_head(x)


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
    print(f"  logits : {tuple(logits.shape)}  (expected (2, 16, {cfg.vocab_size}))")
    print(f"  loss   : {loss.item():.4f}  (~ln(vocab)={math.log(cfg.vocab_size):.4f} at init)")
