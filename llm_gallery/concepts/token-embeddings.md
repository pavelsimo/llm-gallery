---
title: Token embeddings
emoji: 🔤
summary: A big lookup table turns each integer token id into a dense vector — the model's first, learned guess at what that token means.
related: learned-positions, lm-head, hyperparameters, attention
---

## The intuition 🔤

Token ids are just arbitrary integers — `"cat"` might be token 5432 and `"dog"` token 91, and those numbers carry zero meaning. So the first thing the model does is give **every word a GPS coordinate in meaning-space** 🗺️: a learned vector where nearby points mean similar things. After training, *cat* and *dog* end up neighbors, *cat* and *carburetor* end up continents apart.

The embedding layer is nothing fancier than **a table with one row per vocabulary entry — a token id is just a row number to look up.**

## How it works

- 🗂️ The table $E$ has shape `(vocab_size, n_embd)` — e.g. GPT-2's is `50257 × 768`, about 38M parameters on its own.
- 👉 "Embedding" a token is pure indexing: token id 5 means *grab row 5*. In PyTorch this is `nn.Embedding`, which is exactly a matrix plus fancy indexing (no matmul needed).
- 🎓 The rows start random and are trained like any other weights — gradients flow back into exactly the rows that were used.
- 🔗 **Weight tying**: the final LM head must map vectors *back* to vocab scores, i.e. a `(n_embd, vocab_size)` matrix — the transpose shape of $E$. GPT-2 and Gemma reuse the same matrix for both (`lm_head.weight = wte.weight`), saving tens of millions of parameters.

$$h_0 = E[t]$$

where $t$ is the token id and $h_0$ is the vector handed to the first transformer block (after positions are added).

## Mini example: 4-word vocab, 3-dim embeddings

Say the vocabulary is `["the", "cat", "sat", "!"]` and $d = 3$:

```text
E = [[ 0.1, -0.3,  0.2],   # row 0: "the"
     [ 0.9,  0.5, -0.1],   # row 1: "cat"
     [ 0.8,  0.4,  0.0],   # row 2: "sat"
     [-0.5,  0.0,  0.7]]   # row 3: "!"
```

Embedding the sentence `"the cat"` → token ids `[0, 1]`:

```text
h0 = E[[0, 1]] = [[ 0.1, -0.3,  0.2],   # "the"
                  [ 0.9,  0.5, -0.1]]   # "cat"
```

No math happened — just two row lookups. Notice *cat* (row 1) and *sat* (row 2) already point in similar directions; training is what pushed them together.

## Why models use it

- 🌈 **Continuous beats categorical** — vectors let the model interpolate and generalize ("kitten" can land between "cat" and "baby"), which one-hot ids never could.
- ⚡ **Lookup is free** — indexing a table is far cheaper than multiplying by a giant one-hot vector, though mathematically identical.
- 🔗 **Tying with the LM head** closes the loop: the same meaning-space is used to read tokens in and to score tokens out.
