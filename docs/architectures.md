# The architecture taxonomy

The gallery lists 78 models, but they are built from a small menu of reusable ideas. This document is
the "family tree": once you understand each idea here, every model file is just a particular
combination of them. Each model file re-implements the pieces it uses inline (nothing is hidden), so
use this page as the map and the model files as the territory.

## Reading order (suggested)

1. **`gpt2-xl`** — the original decoder-only Transformer. Learn this one cold; everything else is a
   diff against it.
2. **`llama3-8b`** — the modern dense recipe: RoPE + GQA + SwiGLU + RMSNorm.
3. **`olmo2-7b`** — QK-Norm and norm *placement* (post-norm).
4. **`gemma3-27b`** — sliding-window vs global attention, sandwich norm, logit soft-capping.
5. **MoE**: a Qwen3 MoE or `gpt-oss` — routing tokens to a few of many experts.
6. **`deepseek-v3`** — Multi-head Latent Attention (MLA) + fine-grained MoE + shared experts.
7. **Beyond attention**: `xlstm-7b`, a Nemotron Mamba hybrid, `qwen3-next` (Gated DeltaNet).

## 1. Normalization

| Idea | Where | What it does |
|------|-------|--------------|
| **LayerNorm** | GPT-2 | center + scale each token vector to zero-mean/unit-var (learned γ, β) |
| **RMSNorm** | almost all modern models | like LayerNorm but no mean-subtraction and no bias — just scale by RMS. Cheaper, works as well |
| **QK-Norm** | OLMo 2, Qwen3, Gemma 3, GLM | apply RMSNorm to the **queries and keys** (per head) before attention — stabilizes attention logits |

**Placement** matters as much as the norm itself:
- **Pre-norm** (GPT-2, Llama): `x = x + sublayer(norm(x))`. The residual stream stays un-normalized.
- **Post-norm** (OLMo 2): normalize *after* the sublayer. Different gradient flow.
- **Sandwich / peri-LN** (Gemma 3): a norm both before *and* after each sublayer.

## 2. Positional information

A pure attention block is permutation-invariant; position must be injected somehow.

- **Learned absolute** (GPT-2): a trainable vector per position, added to the token embedding. Simple,
  but doesn't extrapolate beyond the trained context length.
- **RoPE** (rotary, almost everyone): rotate the query/key vectors by an angle proportional to their
  position. Encodes *relative* position and extrapolates better.
- **NoPE** (SmolLM3, periodically): some layers get *no* positional encoding at all; the causal mask
  alone carries enough order information. Helps length generalization.
- **Decoupled / partial RoPE** (MLA models): RoPE applied to only part of the head dimension.

## 3. Attention variants

The big lever for cost. All are causal (a token sees only the past).

- **MHA** (multi-head, GPT-2): each head has its own Q, K and V.
- **MQA** (multi-query, Gemma 3 270M): many query heads share **one** K/V head. Tiny KV cache.
- **GQA** (grouped-query, Llama 3+, Qwen3, most): a middle ground — groups of query heads share a K/V
  head. The modern default.
- **MLA** (Multi-head Latent Attention, DeepSeek, Kimi, GLM-5): compress K/V into a small **latent**
  vector and reconstruct per head. Shrinks the KV cache dramatically; pairs with decoupled RoPE.
- **Sliding-window vs global** (Gemma 3, gpt-oss, OLMo 3): most layers attend only to a local window;
  a few layers attend globally. Cuts cost while keeping long-range mixing.
- **Attention sinks / bias** (gpt-oss): a learned per-head bias term in the softmax so the model can
  "attend to nothing", which stabilizes long-context attention.
- **Gated attention** (Qwen3-Next, Tiny Aya, Laguna): a learned gate multiplies the attention output.
- **Sparse / compressed attention** (DeepSeek V3.2/V4, Step 3.5): each query attends to a selected
  sparse subset of keys instead of all of them.

## 4. Feed-forward network

- **GELU MLP** (GPT-2): `Linear -> GELU -> Linear`, hidden size ~4x.
- **SwiGLU / GeGLU** (Llama and modern default): a *gated* MLP —
  `(SiLU(W_gate x) * (W_up x)) -> W_down`. Two input projections; usually ~2.7x hidden so the param
  count matches a 4x GELU MLP.

## 5. Mixture-of-Experts (MoE)

Replace the single MLP with **N expert MLPs** and a **router** that sends each token to only the top-k
experts. Total parameters grow huge while compute per token stays small ("sparse").

- **Token-choice top-k routing**: router scores experts; each token uses its best k.
- **Shared experts** (DeepSeek): one (or few) experts are *always* applied, alongside the routed ones.
- **Fine-grained experts** (DeepSeek): many smaller experts instead of a few big ones.
- **Top-1 routing** (ZAYA1): each token uses exactly one expert.
- **Latent MoE** (Nemotron Super): experts operate in a compressed latent space.
- *Load-balancing aux loss* keeps experts evenly used during training (noted in files; optional here
  since we train tiny demos).

## 6. Beyond attention (sequence mixers)

Some models replace or interleave attention with sub-quadratic mixers:

- **mLSTM / xLSTM**: a modern, parallelizable LSTM variant; fully recurrent, no self-attention.
- **Mamba-2 (SSM)** (Nemotron hybrids): a state-space model layer interleaved with attention/MoE.
- **Gated DeltaNet** (Qwen3-Next/3.5): a linear-attention variant using the delta update rule + gates.
- **Linear / Lightning attention** (Kimi Linear, Ling): attention reformulated to be linear in
  sequence length, usually interleaved with a few full-attention (MLA) layers.

## 7. Block structure & misc

- **Sequential block** (standard): `x -> attn -> ffn` in series.
- **Parallel block** (Tiny Aya, GPT-J style): attention and FFN both read the *same* normalized input
  and their outputs are summed — one less sequential dependency.
- **Weight tying**: share the input embedding matrix with the output (LM head) projection.
- **Embedding scaling** (Gemma): multiply embeddings by √d_model.
- **Logit soft-capping** (Gemma 2/3): `cap * tanh(logits / cap)` to bound pre-softmax logits.

---

Each model file's top docstring states exactly which of the above it uses and links its tech report.
Run `python -m llm_gallery.cli info <slug>` to read it from the terminal.
