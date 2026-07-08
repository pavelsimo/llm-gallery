---
title: Shared experts
emoji: 🤲
summary: One or more experts that every token always passes through, added alongside the routed top-k experts.
related: moe, moe-router, mlp
---

## The intuition 🤲

In a hospital, you don't send every patient straight to a brain surgeon. First everyone sees a **general practitioner** 🩺 who handles the common stuff — blood pressure, basic checks — and *then* the specialists focus purely on what makes each case unusual.

A shared expert is that GP: an expert MLP that **every token visits, always**, with no routing decision, running alongside the routed specialists. DeepSeek-MoE introduced the pattern, and it's now standard in DeepSeek-V3, Qwen3-MoE, and others.

## How it works

- 🩺 One (or a few) **shared experts** process every single token unconditionally — no router, no gate score needed.
- 🚦 The usual **routed experts** still work as normal: the router picks the top-k specialists per token.
- ➕ The outputs are simply **added together** — shared output plus the weighted sum of routed outputs.
- 🧠 **Common knowledge concentrates** in the shared expert (grammar, frequent words, basic patterns), so routed experts stop wasting capacity re-learning the same basics 64 times over and can specialize harder.
- 🧘 It also **stabilizes training**: even if routing is noisy early on, every token is guaranteed at least one well-trained path through the layer.

$$y = E_{shared}(x) + \sum_{i \in \mathrm{TopK}} g_i\, E_i(x)$$

## Mini example: the compute budget

Take a layer with 1 shared expert + 8 routed experts (top-2), each expert 1B parameters:

```text
stored:  1B shared + 8 × 1B routed          = 9B params total

active per token:
         1B shared   (always on)
       + 2B routed   (top-2 of 8)
       = 3B active                          # 3 experts' worth of compute
```

Compared to plain top-2-of-8 MoE, you pay one extra expert of compute per token — but redundancy across the 8 routed experts drops, so each of their parameter budgets goes further. DeepSeek push this by making experts small and numerous (fine-grained), e.g. 1 shared + 8 routed *of 256*.

## Why models use it

- 🎯 **Sharper specialization** — with the common ground absorbed centrally, routed experts differentiate more cleanly (this was DeepSeek-MoE's core argument).
- 🧯 **Routing safety net** — a bad router decision can't strand a token; the shared path always contributes.
- 📦 **Better parameter efficiency** — shared knowledge is stored once instead of duplicated in every expert.
- 👀 In code: a `shared_expert` (or `shared_experts`) MLP whose output is added to the routed sum, sometimes with its own learned sigmoid gate (Qwen-style).
