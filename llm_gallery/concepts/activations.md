---
title: Activation functions
emoji: ⚡
summary: The small nonlinear functions squeezed between linear layers — without them, a deep network would collapse into one big matrix multiply.
related: mlp, glu-feedforward, transformer-block
---

## The intuition ⚡

Stack ten linear layers with nothing in between and you've built... one linear layer — matrix multiplies compose into a single matrix. The activation function is the tiny nonlinear kink that breaks this collapse and lets depth actually *mean* something.

ReLU is a hard **on/off switch** 🔌: negative in, exactly zero out. GELU and SiLU are **dimmer switches** 🎚️: they fade smoothly through zero instead of snapping, even dipping slightly negative before settling — small inputs get *mostly* blocked rather than executed.

## How it works

- 🔌 **ReLU**: $\max(0, x)$ — dead simple, but its gradient is exactly zero for all negative inputs, so a neuron stuck in negative territory stops learning ("dying ReLU").
- 🎚️ **GELU** (GPT-2, BERT): smoothly weights the input by how "positive-ish" it is under a Gaussian. GPT-2 uses the famous tanh approximation — you'll see it verbatim in `gelu_new`:

$$\mathrm{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715x^3)\right]\right)$$

- 🌊 **SiLU / Swish** (Llama, inside SwiGLU): the input times its own sigmoid — self-gating in one line:

$$\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

- 📍 All of these apply **elementwise** — each number in the hidden vector is squashed independently; no mixing across features or tokens.

## Mini example: three activations, five inputs

```text
   x   |  ReLU  |  GELU  |  SiLU
 ------+--------+--------+--------
  -2.0 |  0.00  | -0.05  | -0.24
  -1.0 |  0.00  | -0.16  | -0.27
   0.0 |  0.00  |  0.00  |  0.00
   1.0 |  1.00  |  0.84  |  0.73
   2.0 |  2.00  |  1.95  |  1.76
```

Notice the shape: ReLU slams negatives to zero. GELU and SiLU let a little negative signal *leak through* (a small dip below zero), then converge to ReLU for large positive inputs. That dip is the dimmer in action.

## Why models use it

- 🧮 **Nonlinearity is the whole point** — it's what lets stacked layers represent functions no single linear map can.
- 🌱 **Smooth trains better** — GELU/SiLU have nonzero gradients almost everywhere, so no neurons flatline the way dying ReLUs do; empirically this yields better losses at scale.
- 🏷️ **Spotting them in code**: `F.gelu(x, approximate="tanh")` → GPT-2 lineage; `F.silu(gate)` next to an elementwise multiply → a SwiGLU block (Llama lineage).
- 💸 They're nearly free — a handful of elementwise ops, invisible next to the matrix multiplies they separate.
