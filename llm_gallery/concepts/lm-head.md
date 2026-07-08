---
title: Final norm & LM head
emoji: 🎯
summary: One last normalization, then a single Linear layer turns each hidden state into a score for every word in the vocabulary.
related: decoder-model, token-embeddings, layernorm, hyperparameters
---

## The intuition 🎯

After the last transformer block, each token holds a rich $d$-dimensional vector — but the model must ultimately answer one question: **which token comes next?**

Picture a game-show finale 🎪: the contestant (the hidden state) stands before a wall of 262,144 answer cards (Gemma's vocabulary) and must point at one. The LM head *is* that pointing: it dot-products the hidden vector against a learned direction for every vocabulary word. Words whose directions align with the hidden state score high; softmax turns the scores into a probability over the whole wall.

The final norm is the deep breath before pointing — it rescales the hidden state so it arrives at the head in the range the head was trained for.

## How it works

- 🧼 **Final norm** (`ln_f` in GPT-2, `model.norm` in Llama): one last LayerNorm/RMSNorm on the residual stream. Without it, the stream's magnitude — grown across dozens of additive layers — would skew the logits.
- 🎯 **LM head**: a Linear (no bias, usually) of shape $[d, |V|]$; each vocab word owns one column, and its logit is just a dot product with the hidden state.
- 🔗 **Weight tying**: the head can *reuse the token-embedding matrix transposed* — one matrix serves as dictionary at the entrance and scoreboard at the exit:

$$\mathrm{logits} = h\, W_E^\top$$

- ✂️ Tying saves $|V| \times d$ parameters — for GPT-2 that's 38M of its 124M (~31%!). GPT-2 and Gemma tie; many modern large models (Llama 2/3 at big sizes, DeepSeek) **untie**, since at 70B+ the savings are negligible and separate matrices give a little extra freedom.
- 🌡️ At generation time the logits feed softmax (often after temperature scaling / top-p filtering); at training time they feed cross-entropy directly.

## Mini example: 3 dims, 4-word vocab

Hidden state after the final norm, and a head matrix with one row per word:

```text
h = [1.0, 0.0, 2.0]

W (vocab x d):  "cat"  [ 1.0,  0.0,  1.0 ]
                "dog"  [ 1.0,  0.0,  0.0 ]
                "sat"  [ 0.0,  1.0, -1.0 ]
                "the"  [ 0.5,  0.5,  0.5 ]
```

Logits are dot products, then softmax:

```text
logits = [1+0+2, 1+0+0, 0+0-2, 0.5+0+1] = [3.0, 1.0, -2.0, 1.5]
exp    = [20.09, 2.72, 0.14, 4.48]        sum = 27.42
probs  = [ 0.73, 0.10, 0.005, 0.16]
            cat   dog   sat    the        (cat wins)
```

The hidden state pointed most strongly along "cat"'s direction, so *cat* gets ~73% of the probability mass.

## Why models use it

- 🚪 **The only exit** — every capability the network has must ultimately be expressed through this one Linear; it defines the interface between "thought space" and actual tokens.
- 💰 **It's huge** — $|V| \times d$ is often the single largest matrix in a small model, which is exactly why weight tying was invented and why vocab size is a serious design choice.
- 🔗 **Tying has a logic** — a word's input embedding and its output direction encode related meaning, so sharing them is a sensible prior (and a regularizer) when parameters are scarce.
- 🔍 **In code**: check whether `lm_head.weight is model.embed_tokens.weight` (or a `tie_word_embeddings` config flag) — the quickest way to tell a tied model from an untied one.
