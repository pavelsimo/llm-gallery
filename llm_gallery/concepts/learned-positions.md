---
title: Learned positional embeddings
emoji: 📍
summary: GPT-2 style position encoding — a second lookup table, indexed by position instead of token id, added onto the token embeddings.
related: token-embeddings, rope, attention
---

## The intuition 📍

Attention has a blind spot: it's **permutation-invariant**. Queries and keys are compared as an unordered bag, so without extra help the model literally cannot tell *"dog bites man"* from *"man bites dog"* 🐕. The tokens are the same; only the order differs.

GPT-2's fix is charmingly blunt: keep a second embedding table, one row per *position*, and **stamp every token's vector with a "you are here" 📍 marker before it enters the network.** Row 0 means "first token", row 1 means "second token", and so on.

## How it works

- 🗂️ A learned table `wpe` of shape `(context_length, n_embd)` sits next to the token table `wte` — for GPT-2, `1024 × 768`.
- ➕ The two lookups are simply **added elementwise**: token meaning and position share the same vector space.
- 🎓 The position rows start random and are trained like any other weights — the model *discovers* whatever positional structure is useful, rather than being given a formula.
- 🧱 The catch: there is no row 1025. A model trained with 1,024 positions has learned nothing about longer sequences — the context length is baked into the parameter shapes.

$$h_0 = E_{\mathrm{tok}}[t] + E_{\mathrm{pos}}[p]$$

where $t$ is the token id and $p$ is the token's index in the sequence.

## Mini example: stamping positions onto tokens

Sentence `"the cat"`, $d = 3$:

```text
tok = [[ 0.1, -0.3,  0.2],     pos = [[ 0.0,  0.1,  0.0],   # position 0
       [ 0.9,  0.5, -0.1]]           [ 0.2,  0.0, -0.2]]    # position 1
```

Elementwise add:

```text
h0 = [[ 0.1, -0.2,  0.2],   # "the" @ position 0
      [ 1.1,  0.5, -0.3]]   # "cat" @ position 1
```

If `"cat"` had appeared at position 0 instead, it would carry a *different* vector — same word, different stamp. That difference is the only thing that lets attention distinguish orderings.

## Why models use it

- 🪶 **Dead simple** — one extra `nn.Embedding` and a `+`; no changes to attention itself.
- 🎓 **Fully learned** — no hand-designed sinusoids; the model picks its own encoding.
- ⚠️ **Why newer models moved on**: absolute positions don't extrapolate past the training length and don't directly encode *relative* distance ("3 tokens apart"). RoPE 🌀 fixes both by rotating queries and keys instead of adding a table — which is why Llama, Gemma, and DeepSeek have no `wpe` at all.
