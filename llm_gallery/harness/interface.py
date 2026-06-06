"""The contract every model file conforms to, plus a few shared helpers.

Each file in ``llm_gallery/models`` is fully self-contained, but to let the shared harness
(trainer, sampler, tests) drive *any* of them, they all expose the same tiny surface:

1. A ``Config`` dataclass with (at minimum) ``vocab_size`` and ``context_length`` fields.
2. A module-level ``PRESETS: dict[str, Config]`` mapping names (always including ``"tiny"``) to configs.
3. A ``Model(nn.Module)`` whose ``forward(idx) -> logits`` maps token ids ``[B, T]`` to next-token
   logits ``[B, T, vocab_size]``.

Crucially, the model is *only the architecture*: it returns logits and nothing else. Loss,
optimization, and sampling all live in the harness, so you can study the architecture and the
training mechanics independently. This also keeps the contract identical across all 78 models.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class GalleryConfig(Protocol):
    """Minimum fields the harness reads off any model's config."""

    vocab_size: int
    context_length: int


@runtime_checkable
class GalleryModel(Protocol):
    """The single method the harness calls. ``idx`` is ``[B, T]`` longs; returns ``[B, T, vocab]``."""

    def forward(self, idx: torch.Tensor) -> torch.Tensor: ...


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters.

    ``nn.Module.parameters()`` already de-duplicates tied weights (e.g. when the LM head shares the
    token-embedding matrix), so a plain sum is correct for tied and untied models alike.
    """
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


def human(n: int) -> str:
    """Format a parameter count, e.g. 124_000_000 -> '124.0M'."""
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.1f}{unit}"
    return str(n)


def model_device(model: nn.Module) -> torch.device:
    """Device of the first parameter (all parameters of a model live on one device here)."""
    return next(model.parameters()).device
