"""Autoregressive text sampling for any model conforming to the gallery contract.

Given a prompt of token ids, repeatedly: run the model, take the last position's logits, optionally
apply temperature / top-k / top-p filtering, then sample (or take argmax) the next token and append it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the ``top_k`` highest logits per row; set the rest to -inf."""
    k = min(top_k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[:, [-1]]  # [B, 1] the k-th largest value per row
    return logits.masked_fill(logits < kth, float("-inf"))


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose probabilities sum to >= top_p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Drop tokens once the running probability has already passed top_p, but always keep the top one.
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend ``idx`` ([B, T] longs) by ``max_new_tokens`` sampled tokens; returns [B, T + new]."""
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]  # crop to the model's context window
        logits = model(idx_cond)[:, -1, :]  # [B, V] logits at the final position

        if temperature == 0.0:  # greedy decoding
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                logits = _top_k_filter(logits, top_k)
            if top_p is not None:
                logits = _top_p_filter(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)

        idx = torch.cat([idx, next_id], dim=1)

    if was_training:
        model.train()
    return idx
