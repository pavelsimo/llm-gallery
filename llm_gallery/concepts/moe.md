---
title: Mixture of Experts
emoji: 🧑‍🍳
summary: Replace the single MLP with a bench of expert MLPs and activate only a few per token — huge capacity, small per-token cost.
related: moe-router, shared-experts, mlp, transformer-block
---

## The intuition 🧑‍🍳

Imagine a restaurant with **64 specialist chefs** — one great at pastry, one at sauces, one at grilling. When an order comes in, it doesn't go to every chef; the maître d' routes each dish to the **2 best-suited specialists**, and only they cook.

That's a Mixture of Experts: the transformer block keeps its attention layer, but the single MLP is replaced by many parallel expert MLPs. Each token is sent to just a few of them. The kitchen's total skill is enormous, but each dish only pays for two chefs' time.

## How it works

- 🍽️ **Many experts**: instead of one MLP, the layer holds $N$ independent MLPs (often gated SwiGLU blocks), typically 8–256 of them.
- 🚦 A tiny **router** scores every expert for the current token and picks the **top-k** (commonly $k = 2$, or up to 8 in fine-grained designs like DeepSeek).
- ➕ The chosen experts each process the token, and their outputs are **combined as a weighted sum** using the router's (renormalized) scores.
- 😴 The unchosen experts do nothing for this token — their weights sit in memory but burn no FLOPs. That's the "sparse" in sparse MoE.
- 🎯 Every token can pick a *different* set of experts, so specialization emerges: some experts drift toward code, others toward punctuation or arithmetic.

$$y = \sum_{i \in \mathrm{TopK}} g_i \cdot E_i(x)$$

where $E_i$ is the $i$-th expert MLP and $g_i$ its gate weight from the router.

## Mini example: capacity vs compute

Say each expert MLP has 1B parameters, and the layer has 8 experts with top-2 routing:

```text
dense MLP:    1B params stored,  1B params used per token
MoE (8 × 1B): 8B params stored,  2 experts × 1B = 2B used per token

capacity:  8×  the dense model
compute:  ~2×  the dense model   (plus a tiny router)
```

Eight times the knowledge storage for roughly twice the per-token compute. This is why MoE model names quote two numbers — e.g. **"Qwen3-30B-A3B"** means 30B *total* parameters but only ~3B *active* per token. You pay memory for all of them; you pay compute only for the active ones.

## Why models use it

- 💰 **Decoupled knowledge and compute** — scaling laws reward more parameters, but inference cost tracks *active* parameters; MoE lets you grow one without the other.
- 🧠 Since MLPs are believed to store facts, more experts ≈ **more shelf space for knowledge** without slowing generation.
- 🏭 Used by Mixtral, DeepSeek-V3, Qwen3-MoE, and (reportedly) most frontier models.
- ⚠️ The catch: all experts must fit in memory, and routing needs **load balancing** so tokens don't pile onto a few favorites — see the router concept.
