---
title: Gated MLPs: SwiGLU & GeGLU
emoji: 🤝
summary: The modern feed-forward block runs two parallel up-projections and lets one act as an elementwise valve on the other.
related: mlp, activations, transformer-block, moe
---

## The intuition 🤝

Picture cooking with both hands: one hand holds the ingredients 🥕 (the **up** projection), while the other controls the tap 🚰 (the **gate** projection), deciding how much of each ingredient actually flows into the dish.

That's a gated MLP: instead of one up-projection pushed through an activation, the token is projected up **twice in parallel** — one copy carries the content, the other (after a nonlinearity) multiplies it elementwise, feature by feature. Each hidden feature gets its own learned volume knob.

## How it works

- 🔀 The input $x$ goes through **two separate up-projections**: $xW_{gate}$ and $xW_{up}$, both mapping $d$ → hidden dim.
- 🚰 The gate branch passes through an activation — **SiLU** for SwiGLU (Llama, DeepSeek), **GELU** for GeGLU (Gemma) — producing values that act like soft on/off dials.
- ✖️ The two branches are multiplied **elementwise** ($\odot$): the gate scales each hidden feature of the up branch.
- 📉 A final down-projection $W_{down}$ brings the result back to dimension $d$.
- ⚖️ That's **3 weight matrices instead of 2**, so to keep parameter count comparable to a $4d$ vanilla MLP, the hidden dim is usually shrunk to about $\tfrac{8}{3}d$ (Llama rounds this to a hardware-friendly multiple).

$$\mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(xW_{gate}) \odot xW_{up}\big)W_{down}$$

GeGLU is identical with $\mathrm{GELU}$ in place of $\mathrm{SiLU}$.

## Mini example: gating in action

One token, hidden dim 2. Suppose the two projections give:

```text
x @ W_gate = [1.17, 0.19]
x @ W_up   = [2.00, 2.00]
```

Step 1 — SiLU on the gate branch, $\mathrm{SiLU}(z) = z\,\sigma(z)$:

```text
SiLU(1.17) = 1.17 * σ(1.17) = 1.17 * 0.763 ≈ 0.9   # tap mostly open
SiLU(0.19) = 0.19 * σ(0.19) = 0.19 * 0.547 ≈ 0.1   # tap barely open

gate ≈ [0.9, 0.1]
```

Step 2 — elementwise multiply with the up branch:

```text
gated = [0.9, 0.1] ⊙ [2.0, 2.0] = [1.8, 0.2]
```

The first feature flows through almost fully; the second is throttled to a trickle — same content branch, per-feature control.

## Why models use it

- 🏆 **Better quality per FLOP** — at matched compute, gated MLPs consistently beat the vanilla GELU MLP; that's why Llama, Gemma, Mistral, DeepSeek, and most modern models use them.
- 🎛️ **Multiplicative interactions** — the gate lets the network compute products of features directly, which a single activation can't express as cheaply.
- 🤷 The honest story from the SwiGLU paper: *"we offer no explanation... we attribute their success to divine benevolence"* — it simply wins empirically.
- 👀 In code, look for three linears named `gate_proj`, `up_proj`, `down_proj` — the signature of a modern feed-forward block.
