---
title: Gated attention
emoji: 🚪
summary: A learned sigmoid gate scales the attention output element-by-element, letting each token dial down what attention brought back when it isn't useful.
related: attention, glu-feedforward, qkv-projections, transformer-block
---

## The intuition 🚪

Attention is a meeting: every token listens to the room and comes back with a summary. But not everything heard in a meeting deserves equal weight — sometimes the room had nothing useful to say, and the best move is to **turn the volume down** 🔉.

Gated attention bolts a **volume knob onto each head's output**. The knob setting isn't fixed: it's computed from the token's own hidden state by a small linear layer squashed through a sigmoid, so it always lands between 0 (mute 🔇) and 1 (full volume 🔊). The token itself decides how much of what attention heard actually gets through the door.

## How it works

- 🚪 A gate projection $W_g$ maps the *input* hidden state $x$ to one gate value per output element (or per head), squashed by a sigmoid into $(0, 1)$.
- ⊙ The attention output is multiplied **element-wise** by the gate before the output projection:

$$y = \left(\mathrm{Attn}(x) \odot \sigma(x W_g)\right) W_o$$

- 🎛️ Because the gate depends on $x$, it's **input-conditioned**: the same head can be loud for one token and muted for the next.
- 🧬 Same trick as GLU feed-forwards (SwiGLU) — multiply a signal by a sigmoid-ish gate — just applied to the attention path instead of the MLP path.
- 🏷️ Seen in Qwen3-Next and NVIDIA's gated-attention work; in code it's an extra Linear next to `q/k/v/o_proj`, often named `g_proj` or `gate_proj`, applied right before `o_proj`.

## Mini example: two heads, one knob each

Suppose two heads return these (2-dim) outputs for a token, and the gate layer produces one value per head:

```text
head_1 out = [4.0, 2.0]     gate_1 = sigmoid(2.2) = 0.90
head_2 out = [3.0, 5.0]     gate_2 = sigmoid(-2.2) = 0.10
```

Apply the gates before concatenating and projecting:

```text
gated_1 = 0.90 * [4.0, 2.0] = [3.6, 1.8]    # passes almost untouched
gated_2 = 0.10 * [3.0, 5.0] = [0.3, 0.5]    # nearly silenced
```

Head 2 attended to *something*, but the model decided its report wasn't relevant here — the gate lets it say "thanks, but no" without needing attention weights themselves to change.

## Why models use it

- 🧘 **Training stability** — gating tames activation spikes flowing out of attention, reducing loss spikes at large scale (a headline result of the gated-attention paper).
- 🕳️ **Attention-sink mitigation** — softmax rows must sum to 1, so heads with nothing useful dump weight on token 0 ("the sink"); a gate offers a cleaner escape hatch: just output ~0.
- 🎚️ **Selective by design** — the model gains an explicit "this sublayer is optional" switch per token, per head, learned end-to-end.
- 💸 **Nearly free** — one extra Linear per attention block, a tiny fraction of parameters and FLOPs.
