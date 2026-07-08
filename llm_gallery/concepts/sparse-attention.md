---
title: Sparse attention
emoji: 🕸️
summary: Instead of scoring every token against every other token, each query attends only to a chosen subset — sliding windows, selected blocks, or top-k picks.
related: attention, gqa, mla, linear-attention
---

## The intuition 🕸️

When you look something up in a book you've read, you don't reread every page. You flip to the **sticky notes** 📑 you left on the important ones, plus skim the last few pages you just read.

Full attention rereads everything: every token scores every earlier token, a $T \times T$ bill that explodes at long context. Sparse attention gives each token sticky notes instead: **a rule (or a learned selector) picks a small set of positions worth looking at, and only those pairs are scored.** Everything else is masked out and never computed.

## How it works

- 🪟 **Sliding window**: each token attends only to the last $w$ tokens (Mistral, Gemma's local layers). Compute per token becomes $O(w)$ instead of $O(T)$; stacking layers still lets information travel far, one hop at a time.
- 🧱 **Block / strided patterns**: fixed geometric masks — attend to your own block, plus a few "global" anchor tokens or every $k$-th position (Longformer, BigBird style).
- 🎯 **Learned selection**: a lightweight scorer — a "lightning indexer" in DeepSeek's sparse attention — cheaply rates all past tokens, keeps the **top-k**, and full attention runs only over those winners.
- 🔀 **Hybrids**: many models alternate — e.g. 5 sliding-window layers, then 1 full-attention layer — so cheap local layers do most of the work and rare global layers stitch long-range facts together.

For a query at position $i$ with allowed index set $S(i)$:

$$\mathrm{out}_i = \mathrm{softmax}\!\left(\frac{q_i K_{S(i)}^\top}{\sqrt{d}}\right) V_{S(i)}$$

## Mini example: window of 2

Six tokens, causal, sliding window $w = 2$ (attend to yourself and 1 previous token... plus causality). Allowed pairs as a 0/1 mask (rows = queries, cols = keys):

```text
        t1 t2 t3 t4 t5 t6
   t1 [  1  0  0  0  0  0 ]
   t2 [  1  1  0  0  0  0 ]
   t3 [  0  1  1  0  0  0 ]
   t4 [  0  0  1  1  0  0 ]
   t5 [  0  0  0  1  1  0 ]
   t6 [  0  0  0  0  1  1 ]
```

Full causal attention would score 21 pairs; the window scores 11 — and the gap widens fast: at $T = 32{,}768$ with $w = 4096$, full attention scores ~537M pairs, the window ~134M, a 4× cut that keeps growing linearly instead of quadratically.

## Why models use it

- 💥 **Breaks the $O(T^2)$ barrier** — compute and memory grow linearly (window) or near-linearly (top-k) in sequence length, making 100k+ contexts affordable.
- 💾 **Bounded KV-cache** — a sliding window means old K/V entries can be *evicted*: cache size caps at $w$ no matter how long generation runs.
- 🎯 **Matches how relevance works** — most tokens mostly need their neighbors; the rare long-range lookups can be handled by a few global layers or learned top-k selection.
- 🔍 **In code**: look for a banded/blocked mask instead of a full triangular one, a `sliding_window` config field, or an indexer module scoring keys before the real attention call.
