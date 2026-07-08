---
title: Rotary position embeddings
emoji: 🌀
summary: Encode position by rotating query/key vectors by position-dependent angles, so attention scores automatically depend only on relative distance.
related: learned-positions, attention, qkv-projections
---

## The intuition 🌀

Picture every token holding a set of **clock hands** 🕐. At position 0 the hands point straight up; each step forward in the sequence turns them by a fixed angle — position 5's hands have turned five clicks, position 6's have turned six. Now here's the trick: the *angle between* two tokens' hands depends only on how far apart they are. Positions 5 and 6 look exactly like positions 105 and 106.

Since attention scores are dot products — which measure angles between vectors — rotating Q and K this way makes **relative position fall out of the math for free.**

## How it works

- 👫 The head dimension is split into pairs $(x_{2i}, x_{2i+1})$, and each pair is treated as a point in a 2D plane.
- 🔄 At position $m$, pair $i$ is rotated by angle $m\theta_i$. Each pair gets its own base frequency: fast-spinning hands resolve nearby positions, slow ones track long-range order — a positional "clock" with many hands.
- 🤝 When a rotated query at position $m$ dots a rotated key at position $n$, the absolute rotations cancel and only $m - n$ survives — attention sees *relative* distance.
- 🚫 Nothing is added to the residual stream and no table is learned: RoPE is applied to Q and K *inside* each attention layer, right after the QKV projections. Values are left alone.
- 🧑‍💻 In code you'll see the `precompute_rope` / `apply_rope` pattern: cos and sin for every (position, frequency) pair are computed once and cached, then `apply_rope` combines them with a `rotate_half` trick — `x * cos + rotate_half(x) * sin` — instead of building actual rotation matrices.

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}, \qquad \theta_i = 10000^{-2i/d}$$

## Mini example: one pair, positions 0 and 1

Take a 2-dim query $q = [1, 0]$ and the fastest frequency $\theta_0 = 1$ rad ($\cos 1 \approx 0.54$, $\sin 1 \approx 0.84$):

```text
position 0 (rotate by 0 rad):   q' = [1.00, 0.00]    # unchanged
position 1 (rotate by 1 rad):   q' = [0.54, 0.84]
position 2 (rotate by 2 rad):   q' = [-0.42, 0.91]
```

Now dot a query at position 2 with a key $k = [1, 0]$ rotated to position 1:

```text
q'(pos 2) · k'(pos 1) = -0.42×0.54 + 0.91×0.84 ≈ 0.54 = cos(2 - 1)
```

The score is $\cos(1)$ — exactly what you'd get for positions 1 and 0, or 101 and 100. Only the *gap* matters.

## Why models use it

- 📏 **Relative by construction** — "how far back is that token?" is precisely what language needs, and RoPE encodes it without learning anything.
- 🪶 **Zero parameters** — no `wpe` table, no addition to embeddings; just cached cos/sin buffers.
- 🔭 **Stretchable context** — because positions are angles, you can slow the rotation down to fit more tokens: raising the base `theta` (Llama 3 uses 500,000) or NTK/frequency scaling extends context far beyond the training length — impossible with a learned position table.
- 🏆 Adopted nearly everywhere post-GPT-2: Llama, Gemma, Mistral, Qwen, DeepSeek all use RoPE variants.
