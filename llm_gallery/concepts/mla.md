---
title: Multi-head latent attention
emoji: 🗜️
summary: DeepSeek compresses keys and values into a tiny latent vector, caches only that, and decompresses on demand — a KV-cache far smaller than even GQA's.
related: attention, gqa, qkv-projections, rope
---

## The intuition 🗜️

GQA shrinks the KV-cache by keeping *fewer* filing cabinets. Multi-head latent attention (MLA) goes further: **don't store the documents at all — store a zipped archive 🗜️ and unzip it whenever someone needs to read it.**

Each token's keys and values across *all* heads are squeezed through a low-rank bottleneck into one small **latent vector**. Only that latent is cached. When attention runs, an up-projection reconstructs full per-head keys and values from it. The archive is lossy, but the compression matrices are *learned*, so the model learns to keep exactly what matters.

## How it works

- 🗜️ **Down-projection**: each hidden state is compressed to a small latent (DeepSeek-V3: 7168 → 512):

$$c_t^{KV} = W^{DKV} h_t$$

- 📂 **Up-projection**: full multi-head keys and values are reconstructed from the latent when needed:

$$k_t = W^{UK} c_t^{KV}, \qquad v_t = W^{UV} c_t^{KV}$$

- 🎤 Queries get the same treatment ($h_t \to c_t^Q \to q_t$) — that one saves *activation memory during training* rather than cache.
- 🧭 **Decoupled RoPE**: rotary embeddings don't commute with the up-projection, so a small extra "rope part" (e.g. 64 dims) is computed directly from $h_t$, rotated, and **concatenated** alongside the latent-derived part of each key/query. The cache holds the latent **plus** this one shared rope key.
- 🪄 At inference, $W^{UK}$ can be **absorbed** into the query projection (and $W^{UV}$ into the output projection), so attention scores are computed directly against the cached latents — no explicit unzipping tensor ever materializes.

## Mini example: dimension arithmetic

DeepSeek-V3-style numbers: $d_{model} = 7168$, 128 heads, $d_{head} = 128$, rope dims $= 64$. Cache per token per layer:

```text
MHA:  2 * 128 * 128            = 32,768 values
GQA (n_kv=8, d_head=128):
      2 * 8 * 128              =  2,048 values
MLA:  latent 512 + rope key 64 =    576 values
```

MLA's cache is ~57× smaller than MHA and ~3.6× smaller than an 8-group GQA — while still letting all 128 heads reconstruct their *own* distinct keys and values from the shared latent:

```text
h_t [7168] --W_DKV--> c_KV [512]           # this is what gets cached
c_KV [512] --W_UK--> k [128 heads x 128]   # unzipped on demand (or absorbed)
```

## Why models use it

- 💾 **Smallest cache of the big three** — MLA compresses along the *feature* axis instead of dropping heads, so it beats GQA's memory savings without collapsing head diversity.
- 🧠 **Quality holds up** — DeepSeek reports MLA matching or beating full MHA; the low-rank bottleneck acts like a learned, task-aware compressor rather than a blunt cut.
- ⚡ **Long-context serving** — a 57× cache reduction is the difference between fitting 4 requests and 200 requests per GPU at 128k context.
- 🔍 **In code** look for `kv_a_proj` / `kv_b_proj` (down/up) and separate `rope` vs `nope` head-dim slices — the signature of DeepSeek-V2/V3-family models.
