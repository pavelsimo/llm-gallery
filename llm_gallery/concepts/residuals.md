---
title: Residual connections
emoji: 🛝
summary: Every sublayer adds its output onto its input — x + f(x) — so information and gradients ride an unbroken highway from embeddings to logits.
related: transformer-block, layernorm, attention, mlp
---

## The intuition 🛝

Think of editing a shared document with **tracked changes** 📝. Nobody retypes the whole document — each editor proposes a small *diff* on top of what's already there. The original text survives every round of review, and each contribution is a delta: "add this clarification", "sharpen that phrase".

Residual connections make every transformer sublayer work this way: **a layer doesn't replace the token's vector, it computes an update and adds it on.** The running document is called the *residual stream*, and 100+ layers of edits accumulate on it.

## How it works

- ➕ Around every sublayer (attention *and* MLP), the input is added back to the output: `x = x + sublayer(x)`. In code it's literally one `+` — the cheapest important line in the model.
- 🛣️ Because the identity path is untouched, the gradient of the output with respect to the input is $I + \frac{\partial f}{\partial x}$ — there's always a clean "1" for gradients to flow through. No layer can fully block the signal, so vanishing gradients don't compound with depth.
- 🐣 A layer can also start harmless: with $f(x) \approx 0$ at initialization, the block is a no-op, and training grows each layer's contribution from zero — much easier than learning a full rewrite from scratch.
- 🔀 With **pre-norm** placement (GPT-2 onward), normalization happens on the branch, never on the highway — the residual stream itself is never squashed:

$$x_{l+1} = x_l + f(\mathrm{norm}(x_l))$$

Each block applies this twice — once with $f = \text{attention}$, once with $f = \text{MLP}$.

## Mini example: a layer proposes a small diff

A token's vector entering a sublayer, and the update the sublayer computes:

```text
x            = [1.0,  2.0, -1.0]
f(norm(x))   = [0.1, -0.3,  0.2]     # the layer's proposed edit
```

Add, don't replace:

```text
x_next = x + f(norm(x)) = [1.1, 1.7, -0.8]
```

The vector is still recognizably `x` — nudged, not rewritten. Even if this layer had learned nothing useful ($f \approx 0$), the token's information would pass through intact.

## Why models use it

- 🏗️ **Depth becomes possible** — without residuals, stacking dozens of layers makes gradients vanish and training stall; with them, 80–100+ layer models (Llama 3 70B, DeepSeek-V3) train routinely.
- 🧬 **Iterative refinement** — the model builds meaning gradually: early layers add syntax-ish edits, later layers add semantics-ish ones, all on one shared stream.
- 🔬 **A shared workspace** — attention, MLPs, and the final LM head all read from and write to the same residual stream, which is why interpretability work treats it as the model's central "memory bus".
