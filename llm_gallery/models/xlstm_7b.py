"""xLSTM (7B) — NXAI / Sepp Hochreiter's group, 2025. A language model with NO self-attention.

Where every other model in this gallery mixes tokens with attention, xLSTM uses a modernized,
parallelizable LSTM: the **mLSTM** ("matrix LSTM"). Instead of a vector cell state, each head keeps a
*matrix* memory ``C`` of shape [head_dim, head_dim] that accumulates outer products of (value, key),
gated over time. Reading the memory with the current query gives the output:

    C_t = f_t · C_{t-1} + i_t · (v_t kᵀ_t)         # matrix memory: forget then write
    n_t = f_t · n_{t-1} + i_t · k_t                # normalizer state
    h_t = o_t ⊙ (C_t q_t) / max(|n_tᵀ q_t|, 1)     # read, normalize, output-gate

Because it's a recurrence, position is implicit (no RoPE, no positional embeddings). This file
implements the recurrence **sequentially over time** for clarity — easy to read and obviously causal.
The real xLSTM-7B uses a chunked parallel scan for speed (the original xLSTM paper additionally
interleaved sLSTM blocks and a small causal conv; the 7B release dropped both). Matching the released
7B: q/k are projected at half width (``qk_dim_factor 0.5``, making the memory ``C`` rectangular),
gate preactivations are soft-capped at 15, and a per-head RMSNorm is applied to the memory read-out
before the output gate. The input/forget gates use the paper's stabilized exponential form: a
per-head running max ``m`` keeps ``exp(i_tilde)`` and ``exp(f_tilde)`` numerically bounded while
still allowing input writes stronger than a sigmoid gate.

Diagram: https://sebastianraschka.com/llm-architecture-gallery (xLSTM)
Tech report: https://arxiv.org/abs/2503.13427

Self-contained: every module below is defined in THIS file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "xLSTM (7B)"
RELEASE_DATE = "2025-03-17"
GALLERY_URL = "https://sebastianraschka.com/llm-architecture-gallery"
TECH_REPORT_URL = "https://arxiv.org/abs/2503.13427"


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 50304
    context_length: int = 8192
    n_layer: int = 32
    n_embd: int = 4096
    n_head: int = 8
    qk_dim_factor: float = 0.5  # q/k width relative to n_embd (v stays full width)
    gate_soft_cap: float = 15.0  # input/forget gate preactivations are soft-capped
    intermediate_size: int = 10944  # ffn_proj_factor 2.667 * 4096, rounded up to a multiple of 64
    norm_eps: float = 1e-6
    tie_embeddings: bool = False


PRESETS: dict[str, Config] = {
    "tiny": Config(
        vocab_size=65, context_length=128, n_layer=4, n_embd=128, n_head=4, intermediate_size=256
    ),
    "xlstm-7b": Config(),
}
DEFAULT_PRESET = "xlstm-7b"


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


class mLSTM(nn.Module):
    """Matrix-LSTM token mixer (the heart of xLSTM), written as an explicit causal recurrence."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head  # value / read-out head width
        self.qk_head_dim = int(cfg.n_embd * cfg.qk_dim_factor) // cfg.n_head  # q/k at half width
        self.gate_soft_cap = cfg.gate_soft_cap
        self.q_proj = nn.Linear(cfg.n_embd, int(cfg.n_embd * cfg.qk_dim_factor), bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, int(cfg.n_embd * cfg.qk_dim_factor), bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.i_gate = nn.Linear(cfg.n_embd, cfg.n_head)  # input gate (one scalar per head)
        self.f_gate = nn.Linear(cfg.n_embd, cfg.n_head)  # forget gate
        self.o_gate = nn.Linear(cfg.n_embd, cfg.n_embd)  # output gate (per channel)
        self.out_norm = RMSNorm(self.head_dim, cfg.norm_eps)  # per-head norm on the read-out
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        nh, hd, qk = self.n_head, self.head_dim, self.qk_head_dim
        q = self.q_proj(x).view(b, t, nh, qk)
        k = self.k_proj(x).view(b, t, nh, qk) / math.sqrt(qk)  # scale keys
        v = self.v_proj(x).view(b, t, nh, hd)
        cap = self.gate_soft_cap
        i_tilde = cap * torch.tanh(self.i_gate(x) / cap)  # soft-capped exp-gate preactivation
        f_tilde = cap * torch.tanh(self.f_gate(x) / cap)

        c = x.new_zeros(b, nh, hd, qk)  # rectangular matrix memory per head: value_dim x key_dim
        n = x.new_zeros(b, nh, qk)  # normalizer state
        m = x.new_full((b, nh), float("-inf"))  # running max stabilizer for exp gates
        outputs = []
        for s in range(t):
            m_new = torch.maximum(f_tilde[:, s] + m, i_tilde[:, s])
            i_s = torch.exp(i_tilde[:, s] - m_new).view(b, nh, 1, 1)
            f_s = torch.exp(f_tilde[:, s] + m - m_new).view(b, nh, 1, 1)
            m = m_new
            k_s, v_s, q_s = k[:, s], v[:, s], q[:, s]  # k/q: [B, nh, qk], v: [B, nh, hd]
            c = f_s * c + i_s * (v_s.unsqueeze(-1) @ k_s.unsqueeze(-2))  # write outer product
            n = f_s.view(b, nh, 1) * n + i_s.view(b, nh, 1) * k_s
            num = (c @ q_s.unsqueeze(-1)).squeeze(-1)  # [B, nh, hd] read memory with query
            denom = (n * q_s).sum(-1, keepdim=True).abs().clamp(min=1.0)
            outputs.append(num / denom)

        h = self.out_norm(torch.stack(outputs, dim=1))  # [B, T, nh, hd], per-head norm
        h = torch.sigmoid(self.o_gate(x)) * h.reshape(b, t, d)  # output gate
        return self.out_proj(h)


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
    """Pre-norm block with mLSTM token mixer instead of attention, plus SwiGLU FFN."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.mix_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mix = mLSTM(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.mix(self.mix_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class Model(nn.Module):
    """Full language model: embed tokens, run mLSTM blocks (no positional encoding), project to logits."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # No positional encoding: the recurrence is inherently ordered/causal.
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


# --------------------------------------------------------------------------------------------------
# Standalone smoke test: `python llm_gallery/models/xlstm_7b.py`
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(idx)
    F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1)).backward()
    print(f"{MODEL_NAME}  (tiny preset, no attention)")
    print(f"  params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  logits : {tuple(logits.shape)}")
