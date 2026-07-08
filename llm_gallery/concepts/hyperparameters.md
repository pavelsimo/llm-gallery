---
title: Config & hyperparameters
emoji: 🎛️
summary: A handful of numbers in a dataclass — vocab size, layers, heads, width — completely determine the model's shape and parameter count.
related: token-embeddings, attention, mlp, transformer-block
---

## The intuition 🎛️

Think of a model config as a **recipe card taped to a control panel**. The recipe (the architecture code) is the same for GPT-2 small and GPT-2 XL — what changes is the dials: how many layers to stack, how wide each vector is, how many attention heads to run. Turn the dials up and you get a bigger cake from the exact same instructions.

That's why real codebases start with a tiny `@dataclass Config`: **every tensor shape in the model is derived from these few numbers.**

## How it works

- 📖 `vocab_size` — how many distinct token ids exist (GPT-2: 50,257; Llama 3: 128,256). Sets the size of the embedding table and the LM head.
- 🥞 `n_layer` — how many identical transformer blocks are stacked (GPT-2 small: 12; Llama 3 70B: 80).
- 🧢 `n_head` — how many attention heads per layer; the model width is split evenly among them, so `head_dim = n_embd // n_head`.
- ↔️ `n_embd` (a.k.a. `d_model`, `hidden_size`) — the width of every token's vector as it flows through the network. Almost every weight matrix has this as one of its dimensions.
- 🪟 `context_length` (a.k.a. `block_size`, `max_seq_len`) — the maximum number of tokens the model can attend over at once.
- 🎚️ Plus knobs that tweak behavior rather than shape: dropout probability, RoPE `theta`, whether biases exist, number of KV heads for GQA.

$$\text{params} \approx \underbrace{V \cdot d}_{\text{embeddings}} + \underbrace{L \cdot (12 d^2)}_{\text{blocks (attn + MLP)}}$$

where $V$ is vocab size, $d$ the model width, and $L$ the layer count — a back-of-envelope that lands surprisingly close to the advertised parameter counts.

## Mini example: counting parameters from the config

Take a toy config:

```text
vocab_size = 100
n_embd     = 8
n_layer    = 2
n_head     = 2
```

The embedding table alone is one row per token:

```text
embedding params = vocab_size × n_embd = 100 × 8 = 800
```

Each block holds four attention projections ($4 d^2$) and a 4×-wide MLP ($8 d^2$):

```text
per-block params ≈ 12 × n_embd² = 12 × 64 = 768
total            ≈ 800 + 2 × 768 = 2,336
```

Scale the same arithmetic up to GPT-2 small ($V=50257$, $d=768$, $L=12$) and you get $\approx 38.6\text{M}$ embedding + $\approx 85\text{M}$ block parameters — right at its famous 124M.

## Why models use it

- 🧩 **One recipe, many sizes** — a model family (125M → 70B) is usually the *same code* with different config values, which is why papers publish tables of hyperparameters instead of new diagrams.
- 🔍 **Reading configs is reading architecture** — spotting `n_kv_heads < n_head` tells you it's GQA; `rope_theta = 500000` tells you it's tuned for long context.
- 💰 **Shapes drive cost** — memory and FLOPs follow directly from these numbers, so scaling laws are expressed in terms of them.
