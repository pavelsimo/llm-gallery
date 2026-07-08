---
title: The transformer block
emoji: 🧱
summary: The repeating unit of every LLM — normalize, attend, add back; normalize, MLP, add back — stacked dozens of times.
related: attention, mlp, residuals, layernorm
---

## The intuition 🧱

An LLM is an assembly line, and the transformer block is **the station every token passes through** — the same station, duplicated $N$ times down the line (12 in GPT-2 small, 126 in DeepSeek-V3).

Each station runs the same two-step shift:

1. 🗣️ **A meeting** — attention. Tokens compare notes and exchange information across the sequence. This is the *only* place tokens interact.
2. 🧑‍💻 **Desk work** — the MLP. Each token goes back to its desk and processes what it heard, alone, no talking.

Crucially, nothing is overwritten: each step's result is **added onto** the token's running vector (the residual stream), like stapling a memo to a folder rather than replacing its contents.

## How it works

- 🧼 **Norm first**: the input is normalized (LayerNorm or RMSNorm) *before* each sublayer, so attention and MLP always see well-scaled inputs.
- 🗣️ **Attention sublayer**: tokens mix information across positions, and the result is added back through a residual connection.
- 🧑‍💻 **MLP sublayer**: a position-wise feed-forward net transforms each token independently, again added back.

$$x' = x + \mathrm{Attn}(\mathrm{norm}(x))$$

$$x'' = x' + \mathrm{MLP}(\mathrm{norm}(x'))$$

- 🔀 **Pre-norm vs post-norm**: the original transformer normalized *after* the residual add (post-norm); virtually all modern LLMs normalize *before* (pre-norm, as above), which keeps the residual path clean and makes deep stacks trainable without warmup tricks.
- 🥪 **Sandwich norm**: Gemma 2/3 norm both **before and after** each sublayer ($x + \mathrm{norm}(\mathrm{Attn}(\mathrm{norm}(x)))$) — extra stabilization for the output flowing back into the stream.

## Mini example: shapes through one block

A sequence of $T = 5$ tokens with $d_{model} = 4096$ (batch dim omitted):

```text
x                  [5, 4096]
norm(x)            [5, 4096]   # per-token rescale, shape unchanged
Attn(...)          [5, 4096]   # tokens mix across the seq axis
x + Attn           [5, 4096]   # residual add
norm(x')           [5, 4096]
MLP(...)           [5, 4096]   # widens to 14336 inside, comes back out
x' + MLP           [5, 4096]   # residual add -> block output
```

In and out: `[T, d]` → `[T, d]`. That shape-preservation is what lets you stack the block $N$ times — the output of block 17 is a valid input for block 18.

## Why models use it

- 🔁 **One design, repeated** — scaling a model is mostly "more blocks, wider blocks"; the architecture inside barely changes from GPT-2 to Llama 3.
- 🛣️ **Residuals make depth trainable** — gradients flow straight down the additive path, so 100+ layer stacks still learn.
- ⚖️ **Division of labor** — attention moves information *between* tokens; the MLP transforms information *within* each token. Every LLM ability is built by interleaving those two moves.
- 🔍 **In code**: look for a `Block`/`DecoderLayer` class with exactly this skeleton — two norms, an attention module, an MLP, two `+`s.
