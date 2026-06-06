"""GPT-2 XL (1.5B) — OpenAI, 2019.

The original decoder-only Transformer that this whole gallery descends from. Everything here is the
"classic" recipe; later models in the gallery are best understood as swapping individual pieces of it:

    GPT-2 piece                      modern replacement (see other files)
    ---------------------------      ------------------------------------
    learned absolute positions  ->   RoPE (Llama, Qwen, ...) / NoPE (SmolLM3)
    LayerNorm (with bias)       ->   RMSNorm (almost everyone)
    full multi-head attention   ->   GQA / MQA / MLA
    dense GELU MLP              ->   SwiGLU MLP, or sparse Mixture-of-Experts

Key architectural facts:
  * Token embedding + a *learned* positional embedding table (one vector per absolute position).
  * Pre-LayerNorm Transformer blocks: ``x = x + attn(ln1(x)); x = x + mlp(ln2(x))``.
  * Causal multi-head self-attention (every head sees its own full Q/K/V).
  * MLP expands 4x with a (tanh-approx) GELU nonlinearity.
  * Input embedding and output projection (LM head) share one weight matrix (weight tying).

Diagram: https://sebastianraschka.com/llm-architecture-gallery (GPT-2)
Tech report: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

Self-contained: every module below is defined in THIS file. Reading it top-to-bottom is the point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "GPT-2 XL (1.5B)"
RELEASE_DATE = "2019"
GALLERY_URL = "https://sebastianraschka.com/llm-architecture-gallery"
TECH_REPORT_URL = (
    "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
)


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = 50257
    context_length: int = 1024  # max sequence length (size of the positional embedding table)
    n_layer: int = 48
    n_head: int = 25
    n_embd: int = 1600  # model width; must be divisible by n_head
    dropout: float = 0.0
    bias: bool = True  # GPT-2 uses bias terms in all Linear and LayerNorm layers


# "tiny" is the runnable preset used by tests and the training demo. The named GPT-2 sizes below it
# are the real published configurations, kept for reference.
PRESETS: dict[str, Config] = {
    "tiny": Config(vocab_size=65, context_length=128, n_layer=4, n_head=4, n_embd=128),
    "gpt2-small": Config(n_layer=12, n_head=12, n_embd=768),  # 124M
    "gpt2-medium": Config(n_layer=24, n_head=16, n_embd=1024),  # 355M
    "gpt2-large": Config(n_layer=36, n_head=20, n_embd=1280),  # 774M
    "gpt2-xl": Config(n_layer=48, n_head=25, n_embd=1600),  # 1.5B
}
DEFAULT_PRESET = "gpt2-xl"


# --------------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """LayerNorm with an optional bias. (``nn.LayerNorm`` can't disable bias on its own.)

    Normalizes each token vector to zero mean / unit variance, then applies a learned scale (and
    shift). This stabilizes activations so deep stacks of blocks train well.
    """

    def __init__(self, n_embd: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, written out explicitly for clarity.

    Shapes (B=batch, T=sequence length, C=n_embd, nh=n_head, hd=head_dim=C/nh):
      qkv:  [B, T, 3C] -> three [B, T, C] tensors -> each reshaped to [B, nh, T, hd]
      att:  q @ kᵀ / sqrt(hd)  -> [B, nh, T, T], causally masked, softmaxed over the last axis
      out:  att @ v            -> [B, nh, T, hd] -> merged back to [B, T, C] -> output projection
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # One Linear produces Q, K and V together (3x width), then we split.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular causal mask: position t may attend to positions <= t only.
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)  # each [B, T, C]
        # [B, T, C] -> [B, nh, T, hd]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nh, T, T]
        att = att.masked_fill(~self.causal_mask[:T, :T], float("-inf"))  # block the future
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # [B, nh, T, hd]

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads -> [B, T, C]
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, project back."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.act = nn.GELU(approximate="tanh")  # GPT-2 used the tanh approximation of GELU
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    """A pre-norm Transformer block: normalize, sublayer, then add the residual."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embeddings
        self.wpe = nn.Embedding(cfg.context_length, cfg.n_embd)  # learned positional embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: the output projection reuses the input embedding matrix. Saves parameters
        # and ties "which token is this" to "predict this token".
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 scales the residual-path projections by 1/sqrt(2 * n_layer) for stable deep training.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.config.context_length, f"sequence length {T} > context {self.config.context_length}"
        pos = torch.arange(T, device=idx.device)  # [T] absolute positions 0..T-1

        x = self.wte(idx) + self.wpe(pos)  # [B, T, C]: token meaning + position
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # [B, T, vocab_size]


# --------------------------------------------------------------------------------------------------
# Standalone smoke test: `python llm_gallery/models/gpt2_xl.py`
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    model = Model(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    idx = torch.randint(0, cfg.vocab_size, (2, 16))  # [B=2, T=16]
    logits = model(idx)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1))
    loss.backward()

    print(f"{MODEL_NAME}  (tiny preset)")
    print(f"  params : {n_params:,}")
    print(f"  input  : {tuple(idx.shape)}")
    print(f"  logits : {tuple(logits.shape)}  (expected (2, 16, {cfg.vocab_size}))")
    print(f"  loss   : {loss.item():.4f}  (~ln(vocab)={math.log(cfg.vocab_size):.4f} at init)")
