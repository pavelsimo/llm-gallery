---
title: Parallel attention + MLP
emoji: ⛓️
summary: GPT-J-style blocks feed the same normed input to attention and the MLP simultaneously and add both results back, instead of running them one after the other.
related: transformer-block, attention, mlp, residuals
---

## The intuition ⛓️

In a standard block, the MLP is a chef who waits: attention must finish cooking, plate its dish onto the residual stream, and only then does the MLP start from that updated stream.

The parallel block puts **two chefs at the same prep station** 👨‍🍳👩‍🍳: attention and the MLP both read the *same* normed input, cook **at the same time**, and both dishes land on the residual stream together. Nobody waits, and the kitchen only needs one prep station (one norm) instead of two.

## How it works

- 🔀 One norm, two branches: `norm(x)` is computed once and fed to *both* the attention module and the MLP.
- ➕ Both outputs are added to the residual in a single step:

$$x' = x + \mathrm{Attn}(\mathrm{norm}(x)) + \mathrm{MLP}(\mathrm{norm}(x))$$

- ⚠️ The subtle difference from sequential: the MLP no longer sees what attention just wrote — it works from the *pre-attention* state. Information from this layer's attention only reaches an MLP in the *next* layer.
- 🖥️ **Hardware-friendly**: the two branches have no data dependency, so their matmuls can be fused or overlapped, and under tensor parallelism their weights can be sharded and all-reduced together — one communication round instead of two.
- 🏷️ Introduced by GPT-J, adopted by GPT-NeoX, PaLM, Falcon, and (with tweaks) Cohere's Command models.

## Mini example: sequential vs parallel data flow

Sequential (Llama-style) — two norms, MLP depends on attention's output:

```text
a  = Attn(norm1(x))
x' = x + a
m  = MLP(norm2(x'))     # <-- sees x + a
y  = x' + m
```

Parallel (GPT-J-style) — one norm, no dependency between branches:

```text
n  = norm(x)
a  = Attn(n)            # \  these two can run
m  = MLP(n)             # /  at the same time
y  = x + a + m
```

Same input/output shape `[T, d]`, same parameter count give or take one norm — but the parallel version's critical path is one sublayer long instead of two.

## Why models use it

- ⚡ **Better GPU utilization** — overlapping the attention and MLP matmuls (and their TP all-reduces) buys roughly 15% training speedup at scale, per the PaLM report.
- 🧮 **One norm instead of two** — marginally fewer ops and one less sync point per block.
- ⚖️ **The trade** — small models show a slight quality dip (the MLP loses same-layer access to attention's output), but at large scale the gap vanishes; PaLM ran fully parallel at 540B.
- 🔍 **In code**: the giveaway is a single `input_layernorm` whose output feeds both `self_attn` and `mlp`, with a `hidden + attn_out + mlp_out` line at the end.
