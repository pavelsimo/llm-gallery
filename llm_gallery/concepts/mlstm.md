---
title: mLSTM
emoji: 🧠
summary: The xLSTM matrix-memory cell — classic LSTM gating applied to a matrix state that stores key-value outer products, with exponential gates and a parallelizable form.
related: linear-attention, mamba, attention
---

## The intuition 🧠

Picture a **whiteboard** 🖊️: at each step, the **forget gate** decides how much of the old writing to wipe away — erase 10%? erase most of it? — and the **input gate** decides how boldly to write the new note on top. The board itself never grows; it's rewritten in place, token after token.

The mLSTM (from **xLSTM**, 2024) is the LSTM lineage answering the transformer: keep the beloved gates, but upgrade the cell state from a *vector* to a **matrix** that stores key-value pairs — exactly the associative memory that linear attention builds.

## How it works

- 🧮 The cell state $C_t$ is a **matrix**. Each token writes an outer product $v_t k_t^\top$ into it — key-value storage, just like a linear-attention ledger.
- 🚪 **Forget gate** $f_t$ scales the old board before writing; **input gate** $i_t$ scales the strength of the new note. Both are computed from the current input.
- 🔥 The gates use **exponential activation** (not just sigmoid), letting the cell massively amplify important inputs. To keep $\exp$ from overflowing, a running **max-stabilizer** is tracked and factored out in log-space.
- ⚖️ A **normalizer state** $n_t$ accumulates gated keys the same way $C_t$ accumulates values, so reads are scale-invariant — playing the role softmax's denominator plays in attention.
- 🏎️ There's **no state-to-state nonlinearity** (unlike the classic LSTM's tanh loop), so the whole sequence can be trained in a **parallelized** matrix form — the classic LSTM's fatal flaw, fixed.

$$C_t = f_t\, C_{t-1} + i_t\, v_t k_t^\top, \qquad y_t = \frac{C_t\, q_t}{\max(|n_t^\top q_t|,\ 1)}$$

## Mini example: gates on a scalar board

Collapse everything to scalars: writes $v k = 4$ then $v k = 1$, with per-step gates.

```text
t=1:  f=0.5, i=1.0     C = 0.5 * 0    + 1.0 * 4  = 4.0
      n update:        n = 0.5 * 0    + 1.0 * 1  = 1.0

t=2:  f=0.9, i=2.0     C = 0.9 * 4.0  + 2.0 * 1  = 5.6
      n update:        n = 0.9 * 1.0  + 2.0 * 1  = 2.9

read with q=1:         y = 5.6 / max(|2.9|, 1) ≈ 1.93
```

At $t{=}2$ the forget gate kept 90% of the old writing and the input gate wrote the new note at double strength — then the normalizer $n$ rescaled the read so the output stays well-behaved no matter how bold the gates were.

## Why models use it

- 🌳 **A different lineage** — xLSTM asks "how far do LSTMs get with modern scaling?" and mLSTM is the answer: recurrent nets competitive with transformers at moderate scale.
- 💾 **Constant-size state** — like Mamba and linear attention, inference needs no growing KV-cache; memory and per-token cost are flat in sequence length.
- 🔥 **Exponential gating** lets the cell decisively *overwrite* memory when a crucial token arrives, sharpening recall versus plain accumulate-and-decay.
- 👀 In code (xLSTM, Falcon-style hybrids): look for `C`, `n`, and `m` (max-stabilizer) states threaded through the recurrence, and $q, k, v$ projections feeding a gated matrix update.
