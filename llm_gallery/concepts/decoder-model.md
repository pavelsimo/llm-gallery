---
title: Full decoder model
emoji: 🏗️
summary: The whole assembly — embeddings in the lobby, N transformer-block floors, a final norm, and an LM head that outputs next-token logits at the top.
related: transformer-block, token-embeddings, lm-head, hyperparameters
---

## The intuition 🏗️

Every decoder-only LLM — GPT-2, Llama, Gemma, DeepSeek — is the same skyscraper, just taller or wider. Take the tour:

- 🛬 **Lobby**: token IDs walk in and the embedding table hands each one a $d$-dimensional vector — its outfit for the climb.
- 🏢 **Floors 1 through $N$**: identical transformer blocks. On every floor, tokens meet (attention) and then work solo (MLP), each floor adding refinements onto the same running vector.
- 🔭 **Observation deck**: a final norm tidies the vector, and the LM head turns it into a score for *every word in the vocabulary* — the model's bet on what comes next.

The elevator shaft is the **residual stream**: one vector per token riding all the way up, accumulating notes at every floor.

## How it works

- 🔢 **Embed**: `input_ids [T]` index into an embedding table → `[T, d]` (plus position info — learned positions added here in GPT-2, or RoPE applied inside attention in Llama).
- 🧱 **Stack**: a `for block in self.layers:` loop applies $N$ identical blocks; shape stays `[T, d]` throughout.
- 🧼 **Final norm**: one last LayerNorm/RMSNorm on the stream (`ln_f` / `model.norm`).
- 🎯 **LM head**: a Linear from $d$ to `vocab_size` produces logits **at every position simultaneously**:

$$P(t_{i+1} \mid t_{\le i}) = \mathrm{softmax}\!\left(\mathrm{LMHead}\!\left(\mathrm{norm}\!\left(h_i^{(L)}\right)\right)\right)$$

- 🎓 **Training objective**: pure next-token prediction. Logits at position $i$ are compared (cross-entropy) against the *actual* token at position $i{+}1$ — in code, `logits[:-1]` vs `labels[1:]`, the famous **shift-by-one**. Thanks to the causal mask, one forward pass yields $T$ predictions and $T$ losses at once.

## Mini example: shape trace

GPT-2 small ($d = 768$, 12 blocks, vocab 50257), a 5-token prompt:

```text
input_ids        [5]              # e.g. [464, 3797, 3332, 319, 262]
embeddings       [5, 768]         # lookup (+ positions)
block 1..12      [5, 768]         # shape never changes
final norm       [5, 768]
LM head          [5, 50257]       # logits: one score per vocab word, per position
```

To generate, take row 5's logits (the last token's view of everything), softmax, pick a token, append it, and run again with $T = 6$.

## Why models use it

- 🎯 **One objective, endless skills** — predicting the next token on enough text forces the model to internalize grammar, facts, and reasoning; no task-specific heads needed.
- 🔁 **Trivially scalable** — the whole recipe is four hyperparameters: vocab size, $d$, $N$ layers, context length. GPT-2 to Llama 3 405B is mostly turning those knobs.
- ⚡ **Parallel training, sequential inference** — the causal mask lets training score all $T$ positions in one pass; generation then replays the tower one new token at a time (with a KV-cache to skip redone work).
- 🔍 **In code**: this is the top-level `nn.Module` — embedding, `ModuleList` of blocks, final norm, `lm_head` — usually under 40 lines in a minimal implementation.
