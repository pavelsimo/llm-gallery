---
title: Dropout
emoji: 🎲
summary: During training, randomly zero out a fraction of activations so the network can't lean on any single neuron — a classic regularizer that modern LLMs mostly turn off.
related: residuals, attention, mlp, hyperparameters
---

## The intuition 🎲

Imagine coaching a team where, at every practice, a random half of the players stay home 🏀. Nobody can build a play that hinges on one star teammate always being there — everyone has to develop skills that work with whoever shows up. On game day the full roster plays, and the team is more robust for it.

Dropout does this to neurons: **during training, each activation is randomly zeroed with probability $p$, so no feature can become a single point of failure.**

## How it works

- 🪙 Each forward pass, a fresh random mask $m$ is drawn — every element independently survives with probability $1-p$.
- ⚖️ Survivors are scaled up by $\frac{1}{1-p}$ ("inverted dropout") so the *expected* value of each activation is unchanged — that way inference needs no correction.
- 😴 At eval time (`model.eval()`), dropout is the identity function: all neurons play, nothing is scaled.
- 📍 In GPT-2 it appears in three places: on the embeddings, after the attention weights/projection, and after the MLP projection — all controlled by one `drop_rate` config knob.

$$y = \frac{x \odot m}{1-p}, \qquad m_i \sim \mathrm{Bernoulli}(1-p)$$

## Mini example: 4 activations, p = 0.5

Input vector and one sampled mask:

```text
x = [2.0, 4.0, 6.0, 8.0]
m = [1,   0,   1,   0  ]     # each element kept with prob 0.5
```

Mask, then rescale survivors by 1/(1-0.5) = 2:

```text
x ⊙ m           = [2.0, 0.0, 6.0, 0.0]
y = (x ⊙ m) / 0.5 = [4.0, 0.0, 12.0, 0.0]
```

Expected value check: each element is kept half the time at double strength, so on average $y_i = x_i$. The next forward pass draws a *different* mask — that constant reshuffling is the regularization.

## Why models use it

- 🛡️ **Fights overfitting** — with parts of the network missing at random, memorizing the training set exactly becomes much harder; the model must learn redundant, generalizable features.
- 👥 **Implicit ensembling** — every mask defines a slightly different subnetwork; training averages over millions of them.
- 📉 **Why modern LLMs set it to 0**: dropout earns its keep when you loop over a small dataset many times. LLMs train for roughly *one epoch* over trillions of tokens — every batch is new data, so overfitting isn't the bottleneck and dropout just wastes capacity. That's why you'll find `nn.Dropout` in GPT-2 code but not in Llama, Gemma, or DeepSeek (or you'll see it in the config, set to `0.0`).
