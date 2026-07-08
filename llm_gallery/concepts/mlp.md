---
title: Feed-forward MLP
emoji: 🍔
summary: After attention mixes tokens together, each token goes off alone to a two-layer network that expands, thinks, and compresses back.
related: glu-feedforward, activations, transformer-block, moe
---

## The intuition 🍔

Attention was the meeting — everyone shared notes across the table. The MLP is what happens *after* the meeting: **each token retreats to a private think-tank** 🧑‍💻 to process what it just heard, with no peeking at its neighbors.

It's also widely believed to be where the model *stores facts*. Researchers describe MLP layers as **key-value memories**: the first layer's rows act like pattern detectors ("this looks like *the Eiffel Tower is in ___*") and the second layer writes the matching content ("Paris") back into the token.

## How it works

- 📈 **Expand**: a linear layer projects the token from dimension $d$ up to a wider hidden dimension, classically $4d$ (GPT-2: 768 → 3072).
- ⚡ **Nonlinearity**: an activation like GELU is applied elementwise — without it, the two linear layers would collapse into one.
- 📉 **Project back**: a second linear layer compresses the hidden vector back down to $d$ so it can rejoin the residual stream.
- 🙅 **No cross-token mixing**: the exact same weights are applied to every position independently. All token-to-token communication already happened in attention.
- 🏋️ It's the parameter heavyweight of the block — roughly two-thirds of a transformer layer's weights live here.

$$\mathrm{MLP}(x) = \mathrm{GELU}(xW_1 + b_1)W_2 + b_2$$

where $W_1 \in \mathbb{R}^{d \times 4d}$ and $W_2 \in \mathbb{R}^{4d \times d}$.

## Mini example: 2 dims → 4 dims → 2 dims

One token $x = [1, 2]$, biases zero for simplicity:

```text
W1 = [[ 1, 0, -1, 1],       x @ W1 = [1, 2, -1, 3]
      [ 0, 1,  0, 1]]
```

Apply the nonlinearity (using ReLU here to keep the numbers clean):

```text
hidden = [1, 2, 0, 3]        # the -1 gets zeroed out
```

Project back down with $W_2$:

```text
W2 = [[1, 0],
      [0, 1],       hidden @ W2 = [1*1 + 3*1, 2*1 + 3*(-1)]
      [1, 1],                    = [4, -1]
      [1, -1]]
```

The token went in as $[1, 2]$ and comes out as $[4, -1]$ — transformed entirely on its own, using knowledge baked into the weights.

## Why models use it

- 🧠 **Knowledge storage** — evidence suggests most factual recall (capitals, dates, word meanings) is encoded in these expand-and-compress weights.
- 🔀 **Nonlinear feature mixing** — attention is (mostly) linear blending; the MLP is where genuinely new nonlinear features get computed.
- ⚡ **Embarrassingly parallel** — same weights per token, no sequence dependency, so it maps perfectly onto GPU matrix multiplies.
- 🔧 Modern models (Llama, Gemma, DeepSeek) swap this classic form for **gated variants** like SwiGLU — same role, better quality per FLOP.
