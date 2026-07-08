---
title: RMSNorm
emoji: 📏
summary: LayerNorm on a diet — skip the mean subtraction and bias, just divide by the root-mean-square and apply a learned scale.
related: layernorm, qk-norm, transformer-block
---

## The intuition 📏

Researchers asked a simple question about LayerNorm: **which part actually does the work?** It turns out the magic is almost entirely in the *re-scaling* — forcing vectors to a standard length. The re-*centering* (subtracting the mean) barely matters.

So RMSNorm keeps only the essential move: measure the vector's typical magnitude (its root-mean-square) and divide by it. **Same stabilizing effect, fewer operations, one less learned parameter vector.**

## How it works

- 📏 Compute the RMS of the token's vector: square every element, average, square-root. That's the vector's "typical magnitude".
- ➗ Divide the whole vector by it — the result always has RMS ≈ 1, regardless of input scale.
- 🎛️ Multiply by a learned per-feature weight $\gamma$ (there is **no bias** $\beta$ and **no mean subtraction** — the two deletions relative to LayerNorm).
- 🐭 Cost: one fewer reduction pass over the vector and half the learned parameters of LayerNorm — small per call, but it runs 2–4 times in *every* block of an 80-layer model.
- ⚠️ **Gemma's twist**: Gemma stores its weight as $(1 + w)$ with $w$ initialized to 0, so a fresh layer starts as the identity scale. Porting checkpoints between codebases, `weight` vs `1 + weight` is a classic gotcha.

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

## Mini example: RMS of [2, 4, 6]

Same vector as the LayerNorm example, with $\gamma = 1$:

```text
x = [2.0, 4.0, 6.0]

mean of squares = (4 + 16 + 36) / 3 = 18.667
rms             = √18.667           ≈ 4.32
```

Divide through:

```text
y = [2/4.32, 4/4.32, 6/4.32]
  = [0.46, 0.93, 1.39]
```

Compare LayerNorm's output for the same input, `[-1.22, 0.00, 1.22]`: RMSNorm's result is *not* centered at zero — the mean survives. Only the overall magnitude has been standardized.

## Why models use it

- ⚡ **Cheaper** — no mean pass, no bias add; at billions of tokens per training run these small savings compound.
- 🤷 **Just as good** — empirically, dropping re-centering doesn't hurt quality; the scale control was the load-bearing part.
- 🏆 **The modern default** — Llama, Mistral, Gemma, Qwen, and DeepSeek all use RMSNorm; plain LayerNorm now mostly signals "GPT-2-era architecture".
- 🧊 The same operation reappears inside attention as QK-norm, taming query/key magnitudes before the dot product.
