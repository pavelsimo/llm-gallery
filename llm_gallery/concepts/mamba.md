---
title: Mamba / SSM mixer
emoji: 🐍
summary: A selective state-space layer that replaces attention with a fixed-size recurrent state, deciding on the fly what to remember and what to forget.
related: linear-attention, mlstm, attention, transformer-block
---

## The intuition 🐍

Attention is a reader who keeps the **entire transcript** on the desk and rereads it for every new word. Mamba is a diligent note-taker with **one page of notes** 📝: as tokens stream by, it updates the page — jotting down what seems important, erasing what doesn't — and answers every question from the notes alone. It never goes back to reread the conversation.

The page never grows. Whether the context is 100 tokens or 100,000, the state is the same fixed size.

## How it works

- 🔁 At its core is a **linear recurrence**: a hidden state $h_t$ is carried along the sequence, decayed by $A$, and updated with the new input.

$$h_t = \bar{A}\, h_{t-1} + \bar{B}\, x_t, \qquad y_t = C\, h_t$$

- 🎛️ The **"selective"** part is Mamba's key trick: $B$, $C$, and the step size $\Delta$ are **computed from the current input** $x_t$, not fixed. A large $\Delta$ means "this token matters — write it in and decay old notes"; a tiny $\Delta$ means "skim past this one." Classic SSMs (S4) had fixed dynamics and couldn't do this content-based filtering.
- 📐 The $\bar{A}, \bar{B}$ are **discretized** versions of continuous dynamics: $\bar{A} = \exp(\Delta A)$ — which is why $\Delta$ acts like a forget/attend dial.
- 🏎️ Training doesn't have to loop token by token: a hardware-aware **parallel scan** computes the whole recurrence at once on GPU.
- 🧱 In a full Mamba block, this SSM sits between input/output projections with a convolution and gating around it — it slots in exactly where attention would go.

## Mini example: a scalar note-taker, 3 steps

Toy 1-dim state, with decay $\bar{A} = 0.5$, $\bar{B} = 1$, $C = 2$, starting at $h_0 = 0$. Inputs: $x = [4, 0, 1]$.

```text
t=1:  h = 0.5 * 0    + 4   = 4.0      y = 2 * 4.0   = 8.0
t=2:  h = 0.5 * 4.0  + 0   = 2.0      y = 2 * 2.0   = 4.0   # memory fading
t=3:  h = 0.5 * 2.0  + 1   = 2.0      y = 2 * 2.0   = 4.0   # old + new blended
```

The big input at $t{=}1$ echoes into later steps but decays. In real Mamba, the model could *choose* per token to set the decay near 1 (hold the note) or near 0 (wipe the page) — that choice is the selection mechanism.

## Why models use it

- 💸 **O(T) compute, O(1) memory at inference** — no $T \times T$ score matrix, and no KV-cache growing with every generated token; just one fixed-size state.
- 🚀 Generation speed doesn't degrade with context length — a huge win for long documents and long chats.
- 🧪 The trade-off: a fixed-size state can't store *everything*, so exact long-range recall is weaker than attention's — which is why production models (Jamba, Nemotron-H, Granite 4) are usually **hybrids**: mostly Mamba layers with a few attention layers sprinkled in.
