---
title: MoE router & top-k
emoji: 🚦
summary: A tiny linear layer that scores every expert for each token, keeps the top-k, and renormalizes their weights into a gate.
related: moe, shared-experts, mlp
---

## The intuition 🚦

Every mixture of experts needs a **maître d'** 🤵: someone standing at the door who reads each guest in a split second — "you look like a code token, table 3; you're clearly punctuation, table 7" — and assigns them to the right specialists.

The router is that maître d', and it's shockingly small: a single linear layer mapping the token's $d$-dim vector to one score per expert. All the intelligence of "who should handle this token" lives in one $d \times N$ matrix.

## How it works

- 📊 **Score**: the token vector is multiplied by the router matrix $W_r$, giving one logit per expert.
- 🧮 **Gate**: logits become probabilities via **softmax** (Mixtral) or per-expert **sigmoid** scores (DeepSeek-V3).
- 🏆 **Top-k**: only the $k$ highest-scoring experts are kept — everything else is dropped, which is what makes the layer sparse.
- ⚖️ **Renormalize**: the surviving $k$ gate values are rescaled to sum to 1, so the expert mix stays a proper weighted average.
- 🚨 **Load balancing**: left alone, routers collapse — every token flocks to the same star chef 🌟, that expert overtrains, the rest atrophy, and GPUs holding idle experts sit wasted. Fixes: an **auxiliary loss** nudging expert usage toward uniform (Switch, Mixtral), or **learned bias terms** added to the scores only for selection (DeepSeek-V3's aux-loss-free trick).

$$g = \mathrm{softmax}(x W_r), \quad \text{keep top-}k, \quad g_i \leftarrow \frac{g_i}{\sum_{j \in \mathrm{TopK}} g_j}$$

## Mini example: 4 experts, top-2

A token produces router logits over 4 experts:

```text
logits  = [2.0, 1.0, 0.5, -1.0]
```

Step 1 — softmax:

```text
softmax = [0.61, 0.22, 0.14, 0.03]
```

Step 2 — keep the top-2 (experts 0 and 1), drop the rest:

```text
kept    = [0.61, 0.22]           sum = 0.83
```

Step 3 — renormalize so the weights sum to 1:

```text
gates   = [0.61/0.83, 0.22/0.83] = [0.73, 0.27]

output  = 0.73 * E0(x) + 0.27 * E1(x)
```

Expert 0 does most of the cooking; expert 1 seasons.

## Why models use it

- 🪶 **Nearly free** — a $d \times N$ matrix is thousands of times smaller than the experts it directs; routing costs a rounding error of FLOPs.
- 🎓 **Learned specialization** — the router trains jointly with the experts, so "which expert for which token" is discovered, not hand-designed.
- 🧩 **Hard choices, soft weights** — top-k gives discrete sparsity for efficiency, while the renormalized gates keep the output differentiable for the chosen experts.
- 👀 In code: look for `nn.Linear(hidden_size, num_experts, bias=False)` followed by `topk(...)` — that unassuming pair *is* the router.
