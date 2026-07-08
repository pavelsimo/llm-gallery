---
title: QK normalization
emoji: 🧊
summary: Run RMSNorm on queries and keys right before the dot product, so attention logits can never blow up no matter how large activations grow.
related: rmsnorm, attention, qkv-projections
---

## The intuition 🧊

Attention scores are dot products, and dot products have a failure mode: **magnitude**. If queries and keys drift to large norms during training — which they do, especially in big models — the logits explode, softmax saturates into a one-hot spike on a single token, and gradients through it die. Training loss suddenly spikes 📈💥.

QK-norm is a cooling plate 🧊 bolted directly onto the hot spot: **normalize $Q$ and $K$ the moment before they meet**, so the score can only reflect *direction* (what the vectors mean), never runaway *length*.

## How it works

- 📍 After the Q/K/V projections (and typically per head), apply RMSNorm across the head dimension of $Q$ and of $K$ — usually before RoPE is applied. $V$ is left untouched.
- 🎛️ Each norm has its own tiny learned scale, shared across heads — a negligible parameter cost (two vectors of length `head_dim`).
- 🌡️ With both vectors at RMS ≈ 1, each logit is bounded by roughly $\sqrt{d_k}$ (before the usual $1/\sqrt{d_k}$ division) — softmax stays in its soft, trainable regime instead of collapsing to near-argmax.
- 🧯 This prevents **attention entropy collapse**: the pathology where attention distributions become one-hot, updates turn erratic, and large-scale runs diverge. QK-norm is one of the standard fixes that lets labs train big models without loss spikes.
- 🧑‍💻 In code it reads like: `q = self.q_norm(q); k = self.k_norm(k)` sandwiched between the QKV split and `apply_rope`.

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{\mathrm{RMS}(Q)\,\mathrm{RMS}(K)^\top}{\sqrt{d_k}}\right)V$$

## Mini example: taming a huge logit

A 2-dim query and key that have drifted to large norms:

```text
q = [30, 40],  k = [30, 40]

raw logit = q·k / √2 = (900 + 1600) / 1.41 ≈ 1768   # softmax is now a hard one-hot
```

Normalize each to RMS 1 first (RMS of [30, 40] = √((900+1600)/2) ≈ 35.4):

```text
q̂ = [0.85, 1.13],  k̂ = [0.85, 1.13]

logit = q̂·k̂ / √2 = 2.0 / 1.41 ≈ 1.41   # comfortably inside softmax's soft zone
```

Same direction, same relative preferences across keys — but the score dropped from ~1768 to ~1.4. Softmax can now spread probability instead of spiking.

## Why models use it

- 🧗 **Stability at scale** — logit growth gets worse with model size; QK-norm removes a major source of loss spikes and divergence in large runs.
- 🎓 **Higher learning rates survive** — with the explosion path closed off, training tolerates more aggressive optimization.
- 🪶 **Nearly free** — two extra RMSNorms per attention layer, invisible in the FLOP budget.
- 🏆 Found in Gemma 3, OLMo 2, Qwen 3, and many recent vision transformers — a small line of code that quietly became standard.
