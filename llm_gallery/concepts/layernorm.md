---
title: LayerNorm
emoji: ⚖️
summary: Rescale each token's vector to zero mean and unit variance, then apply a learned scale and shift — keeping activations in a healthy range at every layer.
related: rmsnorm, qk-norm, residuals, transformer-block
---

## The intuition ⚖️

Picture a sound engineer mixing a live show 🎚️. Some microphones come in whisper-quiet, others blast at full volume. Before mixing, the engineer normalizes every channel to the same reference level — *then* applies artistic volume choices per channel. Without that first step, the loud channels would drown out everything downstream.

LayerNorm is that engineer, run on **every token's vector, at every layer**: first force each vector to a standard scale, then let learned knobs ($\gamma$, $\beta$) restore whatever loudness the model actually wants.

## How it works

- 📐 For each token independently, compute the mean $\mu$ and variance $\sigma^2$ *across its $d$ features* — no statistics are shared between tokens or batch items (unlike BatchNorm).
- 🎯 Subtract the mean and divide by the standard deviation: the vector now has mean 0 and variance 1, no matter how wild it came in.
- 🎛️ Multiply by a learned per-feature scale $\gamma$ and add a learned shift $\beta$ (both length-$d$ vectors) so the network isn't stuck at exactly unit scale.
- 🛡️ A tiny $\epsilon$ (e.g. $10^{-5}$) inside the square root avoids division by zero.
- 🔀 **Placement matters**: the original transformer put it *after* each sublayer (post-norm); GPT-2 and everything since put it *before* (pre-norm, `x + attn(norm(x))`), which keeps the residual stream untouched and makes deep stacks train stably without warmup tricks.

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta, \qquad \mu = \frac{1}{d}\sum_i x_i, \quad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$$

## Mini example: normalizing [2, 4, 6]

One token's 3-dim vector, with $\gamma = 1$, $\beta = 0$:

```text
x = [2.0, 4.0, 6.0]

mean     μ  = (2 + 4 + 6) / 3            = 4.0
variance σ² = ((-2)² + 0² + 2²) / 3      = 2.667
std      σ  = √2.667                     ≈ 1.633
```

Subtract the mean, divide by the std:

```text
y = [(2-4)/1.633, (4-4)/1.633, (6-4)/1.633]
  = [-1.22, 0.00, 1.22]
```

Feed in `[20, 40, 60]` instead and you get *exactly the same output* — LayerNorm erases the input's scale and offset, which is precisely the point.

## Why models use it

- 🧯 **Stability at depth** — activations flowing through dozens of layers would otherwise drift toward explosion or collapse; renormalizing at each layer keeps gradients well-behaved.
- 🚀 **Faster, more forgiving training** — loss surfaces get smoother, larger learning rates become safe.
- 🎛️ **Learned scale keeps expressivity** — $\gamma$ and $\beta$ mean normalization constrains *statistics*, not what the layer can represent.
- 💡 Its slimmer successor RMSNorm 📏 drops the mean and $\beta$ entirely — see why Llama-family models made that trade.
