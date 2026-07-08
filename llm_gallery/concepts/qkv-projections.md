---
title: Q/K/V & output projections
emoji: 🎬
summary: Four small Linear layers cast each token into query, key, and value roles, then mix the heads' answers back into one vector.
related: attention, gqa, mla, transformer-block
---

## The intuition 🎬

Think of a token's embedding as one versatile actor. Attention needs that actor to play **three different roles** in the same scene: the *interviewer* asking questions (query 🎤), the *label on a folder* saying what it contains (key 🏷️), and the *content inside the folder* that gets handed over (value 📄).

You don't hire three actors — you give the same one three costumes. That's exactly what the Q, K, and V projections are: **three learned linear layers that re-dress the same token vector for three jobs.** A fourth layer, the output projection, is the editing room 🎞️ that splices all the heads' footage back into one coherent cut.

## How it works

- 🎤 `q_proj`: $q = x W_q$ — what this token is *looking for*.
- 🏷️ `k_proj`: $k = x W_k$ — what this token *advertises* to others.
- 📄 `v_proj`: $v = x W_v$ — what this token *hands over* when it's picked.
- 🎞️ `o_proj`: after attention, head outputs are concatenated and multiplied by $W_o$, which lets heads exchange and recombine what they each found.
- 🧵 **Fused vs separate**: GPT-2 uses one big `c_attn` Linear of width $3 d_{\text{model}}$ and splits the result into Q, K, V; Llama-style code keeps three separate `q_proj` / `k_proj` / `v_proj` layers (handy when K/V are smaller, as in GQA).
- 🔀 **Reshaping for heads**: the projected tensor `[batch, seq, d_model]` is `.view()`-ed to `[batch, seq, heads, head_dim]` and `.transpose(1, 2)`-ed to `[batch, heads, seq, head_dim]` — no math, just slicing one wide vector into per-head chunks.

$$q = x W_q, \qquad k = x W_k, \qquad v = x W_v, \qquad y = \mathrm{concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W_o$$

## Mini example: one token, 2 dims

Take a single token vector $x = [1, 2]$ and tiny $2 \times 2$ weight matrices:

```text
W_q = [[1, 0],     W_k = [[0, 1],     W_v = [[ 2, 0],
       [0, 1]]            [1, 0]]            [ 0, 2]]
```

Multiply the same $x$ by each:

```text
q = x @ W_q = [1, 2]      # "what am I seeking?"
k = x @ W_k = [2, 1]      # "what do I advertise?"
v = x @ W_v = [2, 4]      # "what do I hand over?"
```

One input vector, three different views of it — the roles differ only because the learned weights differ. In a real model this happens for every token at once:

```text
x:  [batch, seq, 4096]
q:  [batch, seq, 4096]  --view-->  [batch, seq, 32, 128]  --transpose-->  [batch, 32, seq, 128]
```

## Why models use it

- 🎭 **Decouples the roles** — what a token searches for, what it matches on, and what it contributes can all be learned independently.
- 📐 **Sets up the head split** — the projections produce one wide vector that reshaping cheaply slices into many small per-head subspaces.
- 🎞️ **The output projection is where heads talk** — without $W_o$, heads would stay isolated silos; with it, their findings blend back into the residual stream.
- 🔍 **Spotting them in code** is the fastest way to identify an attention module: `c_attn`/`c_proj` means GPT-2 lineage, `q_proj`/`k_proj`/`v_proj`/`o_proj` means Llama lineage.
