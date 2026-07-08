---
title: Multi-head causal self-attention
emoji: 🔦
summary: Each token shines a flashlight backwards over the sequence and gathers information from the tokens that matter most.
related: qkv-projections, rope, gqa, transformer-block
---

## The intuition 🔦

Imagine you're reading a mystery novel and hit the sentence "she finally opened **it**". To understand *it*, your eyes dart back through earlier pages looking for candidates — the locked box 📦, the letter ✉️, the door 🚪. You weigh each candidate by relevance, blend their meanings, and move on.

That's self-attention: **every token gets to look back at all previous tokens, score how relevant each one is, and pull in a weighted mix of their information.**

## How it works

- 🎭 Each token produces three vectors: a **query** $Q$ ("what am I looking for?"), a **key** $K$ ("what do I contain?"), and a **value** $V$ ("what will I hand over if picked?").
- 🤝 Relevance is measured by dot products between queries and keys — one score for every (token, earlier-token) pair.
- 🌡️ Scores are divided by $\sqrt{d_k}$ so they don't explode as head dimension grows, then squashed into probabilities with a softmax.
- ⛔ The **causal mask** sets all scores for *future* tokens to $-\infty$ before the softmax — a token may only attend to itself and its past (that's what makes generation left-to-right).
- 🧢 **Multi-head**: instead of one big attention, the model runs many small ones in parallel (e.g. 32 heads), each in its own subspace. One head can track syntax, another coreference, another positions. Their outputs are concatenated and mixed by an output projection.

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

where $M$ is the causal mask ($0$ on allowed positions, $-\infty$ on future ones).

## Mini example: 2 tokens, 2 dims

Say the sequence is `["the", "cat"]` and one head has $d_k = 2$:

```text
Q = [[1, 0],      K = [[1, 0],      V = [[10,  0],
     [1, 1]]           [0, 1]]           [ 0, 10]]
```

Step 1 — raw scores $QK^\top / \sqrt{2}$:

```text
scores = [[0.71, 0.00],     # "the" vs {the, cat}
          [0.71, 0.71]]     # "cat" vs {the, cat}
```

Step 2 — causal mask (token 1 can't see token 2):

```text
masked = [[0.71, -inf],
          [0.71, 0.71]]
```

Step 3 — softmax each row, then multiply by $V$:

```text
weights = [[1.00, 0.00],        out = [[10.0, 0.0],   # "the": only itself
           [0.50, 0.50]]              [ 5.0, 5.0]]    # "cat": 50/50 blend
```

The token *"cat"* ends up carrying half of *"the"*'s information — attention literally mixes token representations.

## Why models use it

- 🛣️ **Direct highways between distant tokens** — token 1 and token 4,000 are one matrix multiply apart, so long-range dependencies don't fade like in RNNs.
- ⚡ **Parallel training** — all positions compute at once; no sequential recurrence.
- 💰 The cost: scores form a $T \times T$ matrix, so compute and the KV-cache grow with sequence length — the reason variants like GQA, sliding windows, and latent attention exist.
