---
title: Linear attention
emoji: ➰
summary: Drop the softmax and attention becomes associative — you can pre-sum keys and values into one running state, turning attention into an RNN.
related: attention, mamba, mlstm, gated-attention
---

## The intuition ➰

Standard attention keeps the **full transcript**: every past key and value stays in the KV-cache, and each new token compares against all of them. Linear attention keeps a **running summary ledger** 📒 instead: each past token is folded into one fixed-size matrix, and new queries read from the ledger — never from the raw history.

The trick is a single algebraic move. Softmax attention computes $\mathrm{softmax}(QK^\top)V$, and the softmax welds $Q$ and $K$ together — you *must* build the $T \times T$ score matrix. Remove the softmax and matrix multiplication becomes **associative**:

$$(QK^\top)V = Q(K^\top V)$$

$K^\top V$ is a small $d \times d$ matrix, independent of sequence length — and it can be built incrementally, token by token. Attention just became an RNN.

## How it works

- 📒 The **state** $S$ is a $d \times d$ matrix. Each token deposits its key-value pair into it as an **outer product** $k_t v_t^\top$ — "under address $k_t$, file content $v_t$."
- 🔍 A new token **reads** by multiplying its query against the state: queries similar to a stored key retrieve (a blend of) that key's value.
- 🚪 A **gate** $\alpha_t \in (0,1]$ decays the old state before each write — pure accumulation ($\alpha = 1$) lets the ledger fill with stale entries, so modern variants learn to forget.
- ✍️ **Delta-rule** variants (DeltaNet, GatedDeltaNet) go further: instead of only adding, they first *erase* whatever the state currently holds at key $k_t$, then write the new value — **overwrite, not accumulate**. GatedDeltaNet combines this with the decay gate and powers Qwen3-Next-style hybrids.

$$S_t = \alpha_t S_{t-1} + k_t v_t^\top, \qquad y_t = S_t^\top q_t$$

## Mini example: a 2-dim ledger, 2 tokens

Token 1 writes $k_1 = [1, 0]$, $v_1 = [3, 0]$; token 2 writes $k_2 = [0, 1]$, $v_2 = [0, 5]$. No decay ($\alpha = 1$):

```text
S1 = k1 v1ᵀ = [[3, 0],        S2 = S1 + k2 v2ᵀ = [[3, 0],
               [0, 0]]                             [0, 5]]
```

Now token 2 queries with $q_2 = [1, 0]$ — "what was filed under the first key?":

```text
y2 = S2ᵀ q2 = [[3, 0],ᵀ  [1,   = [3, 0]     # retrieves v1
               [0, 5]]    0]
```

Both tokens' information lives in one 2×2 matrix — the state would stay 2×2 after a million tokens too.

## Why models use it

- 💾 **Constant memory, linear time** — no KV-cache growth; generating token 100,000 costs the same as token 10.
- 🧠 The price: the ledger is **lossy** — cram a long history into $d \times d$ numbers and precise recall of arbitrary old tokens suffers, which softmax attention does effortlessly.
- 🔀 Hence **hybrid stacks**: Kimi Linear and Qwen3-Next-style models interleave many linear-attention layers with occasional full-attention layers — cheap summaries most of the way, exact lookup when it counts.
- 👪 Mamba, mLSTM, and gated linear attention are close cousins — all fixed-size-state recurrences differing mainly in how they gate and normalize the ledger.
