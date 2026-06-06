"""A small, readable training loop that works on any model conforming to the gallery contract.

It deliberately spells out the full mechanics so the training side is as studyable as the models:
batching, forward, cross-entropy on next-token logits, backward, gradient clipping, AdamW with a
cosine learning-rate schedule + warmup, periodic validation, and checkpointing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CharDataset, get_batch


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 32
    block_size: int = 128  # sequence length used for training (<= model.context_length)
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    seed: int = 1337
    device: str = "cpu"


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay from ``lr`` to ``min_lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, cfg: TrainConfig) -> float:
    """Average cross-entropy over a few random batches (used for validation)."""
    was_training = model.training
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    for k in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.batch_size, cfg.block_size, cfg.device)
        logits = model(x)
        losses[k] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    if was_training:
        model.train()
    return losses.mean().item()


def train(
    model: nn.Module,
    dataset: CharDataset,
    cfg: TrainConfig,
    ckpt_path: Path | None = None,
    extra_state: dict | None = None,
) -> list[tuple[int, float, float]]:
    """Train ``model`` on ``dataset``; returns a list of (step, train_loss, val_loss) samples.

    ``extra_state`` (e.g. the model's config + slug) is merged into the saved checkpoint so a model
    can be rebuilt later for generation without guessing its hyperparameters.
    """
    torch.manual_seed(cfg.seed)
    model.to(cfg.device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    history: list[tuple[int, float, float]] = []
    t0 = time.time()
    for step in range(cfg.steps):
        # Manually drive the learning rate (one schedule for every parameter group).
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(dataset.train, cfg.batch_size, cfg.block_size, cfg.device)
        logits = model(x)  # [B, T, V]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == cfg.steps - 1:
            val = estimate_loss(model, dataset.val, cfg)
            history.append((step, loss.item(), val))
            dt = time.time() - t0
            print(
                f"step {step:5d} | train {loss.item():.4f} | val {val:.4f} | "
                f"lr {lr:.2e} | {dt:5.1f}s"
            )

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model.state_dict(), "stoi": dataset.stoi, "itos": dataset.itos}
        if extra_state:
            payload.update(extra_state)
        torch.save(payload, ckpt_path)
        print(f"saved checkpoint -> {ckpt_path}")

    return history
