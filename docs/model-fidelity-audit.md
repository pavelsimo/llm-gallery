# Model Fidelity Audit

Audit date: 2026-07-07

This audit cross-checks the gallery model implementations against the embedded architecture diagrams
under `web/assets/architectures/`, the model tech-report links exposed by each module, and the public
config fixtures in `tests/fixtures/hf_configs/`.

## Summary

- Tier 1 plus the TODO-listed DeepSeek R1 path were checked at architecture-invariant level in
  `tests/test_tier1_architecture_fidelity.py`.
- Tier 2 and tier 3 config fidelity continue to be covered by `tests/test_real_presets.py` and
  `scripts/audit_fidelity.py`; tier 3 files are treated as config/template variants unless their
  module docstring marks an assumption.
- No unambiguous implementation mismatch required a model-code rewrite in this pass.
- One public fixture was added for `xlstm-7b`; the audit mapper now understands xLSTM's public
  config names (`num_blocks`, `num_heads`, `embedding_dim`).

## Audited Architecture Checks

The new invariant tests cover the main facts called out in `TODO.md`:

- `gpt2-xl`: MHA with packed QKV projection, learned absolute positions, LayerNorm, GELU MLP,
  tied input/output embeddings, and GPT-2 XL published dimensions.
- `llama3-8b`: GQA, RoPE, RMSNorm, no-bias projections, SwiGLU, untied output head, and published
  Llama 3 8B dimensions.
- `deepseek-v3` and `deepseek-r1`: MLA low-rank query/KV path, decoupled RoPE dimensions, first
  dense layers followed by DeepSeekMoE, shared experts, top-k routing, and published dimensions.
- `gemma3-27b`: sliding/global layer cadence, sandwich GemmaRMSNorm, per-head QK norm, GeGLU,
  embedding scaling, tied embeddings, decoupled head dimension, and published dimensions.
- `qwen3-next-80b-a3b`: Gated DeltaNet layers, periodic gated-attention layers, partial RoPE,
  depthwise short conv, MoE routing, and published dimensions.
- `kimi-linear`: linear/full-attention hybrid cadence, MoE routing, published outer dimensions, and
  an explicit docstring assumption that the educational implementation simplifies the real MLA/KDA
  internals to plain GQA plus vanilla linear attention.
- `xlstm-7b`: attention-free mLSTM recurrence, stabilized gates, no RoPE/causal-mask buffers,
  SwiGLU FFN, and public xLSTM config fields.
- `nemotron3-nano-30b`: flat `M`/`E`/`*` macro pattern, Mamba depthwise conv/scan, MoE FFN layers,
  periodic GQA attention layers, and published dimensions.

## Intentional Simplifications

- Kimi Linear keeps the published outer dimensions and hybrid cadence, but its full-attention layers
  are plain GQA and its linear mixer is a readable linear-attention recurrence rather than the full
  production KDA/MLA stack.
- xLSTM keeps the mLSTM core and published width/depth/head count, while omitting the production
  sLSTM/causal-conv details already called out in the module docstring.
- Mamba, DeltaNet, linear attention, and mLSTM scans are written sequentially for readability; the
  production papers use chunked or parallel scan kernels.
- Long-tail tier 3 model assumptions remain documented with `ASSUMPTION:` / `NOTE:` markers and are
  surfaced in the generated visualizer notes.

## Fixture Notes

Several high-priority public configs are gated or require authentication from Hugging Face in this
environment, including Llama 3 8B, Gemma 3 27B, and Nemotron 3 Nano. Those models are still checked by
explicit published-field tests and module tech-report links. The existing HF fixture suite covers the
available public config JSON files and now includes xLSTM.
