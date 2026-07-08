---
title: Grouped-query attention
emoji: 👥
summary: Many query heads share a smaller pool of key/value heads, shrinking the KV-cache at inference with almost no quality loss.
related: attention, qkv-projections, mla, hyperparameters
---

## The intuition 👥

Picture 32 detectives 🕵️ working a case. In classic multi-head attention each detective keeps a **private filing cabinet** of evidence — 32 cabinets, mostly full of duplicated paperwork.

Grouped-query attention says: the questions detectives ask are all different, but the evidence they consult overlaps a lot. So give them **8 shared filing cabinets**, 4 detectives per cabinet. Everyone still asks their own question (32 query heads), but keys and values come from a shared, smaller pool (8 KV heads). Storage drops 4×, and the investigation barely suffers.

## How it works

- 🎤 The model keeps the full count of **query** heads (`n_head`, e.g. 32) but projects far fewer **key/value** heads (`n_kv_head`, e.g. 8) — in code, `k_proj` and `v_proj` are simply narrower Linears.
- 👥 Query heads are split into **groups**; each group of $n_{head}/n_{kv}$ query heads attends against the *same* K/V head.
- 💾 The win is at inference: the **KV-cache** stores keys and values for every past token, and its size per layer is

$$\text{cache} = 2 \cdot T \cdot n_{kv} \cdot d_{head}$$

  (the 2 is for K and V). GQA shrinks it by exactly $n_{head} / n_{kv}$.
- 🔁 In code you'll see a `repeat_kv` helper: it `expand`s each KV head $n_{head}/n_{kv}$ times along the head dimension so the shapes line up with Q and the usual attention math runs unchanged.
- 1️⃣ **MQA** (multi-query attention) is the extreme: a single KV head shared by *all* query heads — maximum savings, slightly bigger quality hit. GQA is the sweet spot in between; MHA is the other extreme ($n_{kv} = n_{head}$).

## Mini example: cache arithmetic

Llama-style config: 32 query heads, $d_{head} = 128$, sequence length $T = 8192$, fp16 (2 bytes), one layer.

```text
MHA  (n_kv = 32):  2 * 8192 * 32 * 128 = 67,108,864 values  ->  128 MiB
GQA  (n_kv =  8):  2 * 8192 *  8 * 128 = 16,777,216 values  ->   32 MiB
MQA  (n_kv =  1):  2 * 8192 *  1 * 128 =  2,097,152 values  ->    4 MiB
```

Across 32 layers, MHA needs 4 GiB of cache per sequence; GQA needs 1 GiB — a 4× cut, which directly multiplies how many sequences fit on one GPU. The `repeat_kv` step at compute time:

```text
K: [batch, 8, T, 128]  --repeat_kv(4)-->  [batch, 32, T, 128]   # now matches Q
```

## Why models use it

- 💾 **KV-cache is the inference bottleneck** — at long context it dwarfs activations, so shrinking it 4–8× means bigger batches and longer contexts on the same hardware.
- ⚖️ **Nearly free quality-wise** — queries carry most of the head diversity; sharing K/V loses very little (the GQA paper even converts trained MHA checkpoints by mean-pooling KV heads).
- ⚡ **Faster decoding** — less cache to read per generated token means less memory bandwidth, the real limiter at inference.
- 🏷️ **Everywhere in modern configs** — Llama 2 70B, Llama 3, Mistral, Qwen, Gemma all ship with `n_kv_head < n_head`; spotting the two numbers in a config file tells you the group size instantly.
