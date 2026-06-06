"""Registry of every model in the gallery: a slug -> metadata map used by the CLI and tests.

Modules are imported *lazily* (only when a model is actually built), so listing the gallery is cheap
and a not-yet-implemented entry doesn't break anything. As each model file is finished, flip its
``status`` to ``"done"`` and the test suite will start covering it automatically.

The order mirrors the gallery. ``module`` is the file stem under ``llm_gallery/models``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

DONE = "done"
PLANNED = "planned"


@dataclass(frozen=True)
class Entry:
    slug: str  # CLI / registry key, e.g. "qwen3-0.6b"
    module: str  # file stem under llm_gallery.models, e.g. "qwen3_0_6b"
    name: str  # human-readable name
    release: str  # release date (or "")
    archetype: str  # one-line architecture descriptor
    status: str = PLANNED


# fmt: off
REGISTRY: list[Entry] = [
    Entry("gpt2-xl", "gpt2_xl", "GPT-2 XL (1.5B)", "2019", "MHA + learned absolute positions", DONE),
    Entry("llama3-8b", "llama3_8b", "Llama 3 (8B)", "2024-04-18", "GQA + RoPE + SwiGLU + RMSNorm", DONE),
    Entry("llama3.2-1b", "llama3_2_1b", "Llama 3.2 (1B)", "2024-09-25", "GQA (small)", DONE),
    Entry("olmo2-7b", "olmo2_7b", "OLMo 2 (7B)", "2024-11-25", "MHA + QK-Norm + post-norm", DONE),
    Entry("deepseek-v3", "deepseek_v3", "DeepSeek V3 (671B)", "2024-12-26", "MLA + sparse MoE + shared experts", DONE),
    Entry("deepseek-r1", "deepseek_r1", "DeepSeek R1 (671B)", "2025-01-20", "MLA sparse MoE (reasoning)", DONE),
    Entry("gemma3-27b", "gemma3_27b", "Gemma 3 (27B)", "2025-03-11", "GQA + sliding/global + sandwich norm", DONE),
    Entry("mistral-small-3.1", "mistral_small_3_1", "Mistral Small 3.1 (24B)", "2025-03-18", "GQA dense", DONE),
    Entry("llama4-maverick", "llama4_maverick", "Llama 4 Maverick (400B)", "2025-04-05", "sparse MoE + GQA", DONE),
    Entry("qwen3-0.6b", "qwen3_0_6b", "Qwen3 (0.6B)", "2025-04-28", "GQA + QK-Norm", DONE),
    Entry("qwen3-235b-a22b", "qwen3_235b_a22b", "Qwen3 (235B-A22B)", "2025-04-28", "sparse MoE + GQA + QK-Norm", DONE),
    Entry("qwen3-30b-a3b", "qwen3_30b_a3b", "Qwen3 (30B-A3B)", "2025-04-28", "sparse MoE + GQA", DONE),
    Entry("qwen3-32b", "qwen3_32b", "Qwen3 (32B)", "2025-04-28", "GQA + QK-Norm", DONE),
    Entry("qwen3-4b", "qwen3_4b", "Qwen3 (4B)", "2025-04-28", "GQA + QK-Norm", DONE),
    Entry("qwen3-8b", "qwen3_8b", "Qwen3 (8B)", "2025-04-28", "GQA + QK-Norm", DONE),
    Entry("smollm3-3b", "smollm3_3b", "SmolLM3 (3B)", "2025-06-19", "GQA + periodic NoPE", DONE),
    Entry("kimi-k2", "kimi_k2", "Kimi K2 (1T)", "2025-07-10", "MLA sparse MoE", DONE),
    Entry("glm-4.5", "glm_4_5", "GLM-4.5 (355B)", "2025-07-28", "sparse MoE + GQA + QK-Norm", DONE),
    Entry("gpt-oss-120b", "gpt_oss_120b", "GPT-OSS (120B)", "2025-08-04", "sparse MoE + alternating attn + sinks", DONE),
    Entry("gpt-oss-20b", "gpt_oss_20b", "GPT-OSS (20B)", "2025-08-04", "sparse MoE + alternating attn + sinks", DONE),
    Entry("gemma3-270m", "gemma3_270m", "Gemma 3 (270M)", "2025-08-14", "MQA + sliding/global", DONE),
    Entry("grok-2.5", "grok_2_5", "Grok 2.5 (270B)", "2025-08-22", "sparse MoE + GQA", DONE),
    Entry("qwen3-next-80b-a3b", "qwen3_next_80b_a3b", "Qwen3 Next (80B-A3B)", "2025-09-09", "Gated DeltaNet + gated attn", DONE),
    Entry("minimax-m2", "minimax_m2", "MiniMax M2 (230B)", "2025-10-23", "sparse MoE + GQA + QK-Norm", DONE),
    Entry("kimi-linear", "kimi_linear", "Kimi Linear (48B-A3B)", "2025-10-30", "linear attn + MLA hybrid", DONE),
    Entry("olmo3-32b", "olmo3_32b", "OLMo 3 (32B)", "2025-11-20", "GQA + QK-Norm + sliding/global", DONE),
    Entry("olmo3-7b", "olmo3_7b", "OLMo 3 (7B)", "2025-11-20", "MHA + QK-Norm + sliding/global", DONE),
    Entry("deepseek-v3.2", "deepseek_v3_2", "DeepSeek V3.2 (671B)", "2025-12-01", "MLA + sparse attention + MoE", DONE),
    Entry("mistral-large-3", "mistral_large_3", "Mistral Large 3 (673B)", "2025-12-02", "sparse MoE + MLA", DONE),
    Entry("nemotron3-nano-30b", "nemotron3_nano_30b", "Nemotron 3 Nano (30B-A3B)", "2025-12-04", "Mamba-2 hybrid + MoE", DONE),
    Entry("mimo-v2-flash", "mimo_v2_flash", "Xiaomi MiMo-V2-Flash (309B)", "2025-12-16", "sparse MoE + sliding/global", DONE),
    Entry("glm-4.7", "glm_4_7", "GLM-4.7 (355B)", "2025-12-22", "sparse MoE + GQA + QK-Norm", DONE),
    Entry("arcee-trinity-large", "arcee_trinity_large", "Arcee Trinity Large (400B)", "2026-01-27", "sparse MoE + gated attn + sliding/global", DONE),
    Entry("glm-5", "glm_5", "GLM-5 (744B)", "2026-02-11", "sparse MoE + MLA + sparse attn", DONE),
    Entry("nemotron3-super-120b", "nemotron3_super_120b", "Nemotron 3 Super (120B-A12B)", "2026-03-11", "Mamba-2 hybrid + latent MoE", DONE),
    Entry("gemma4-31b", "gemma4_31b", "Gemma 4 (31B)", "2026-04-02", "GQA + QK-Norm + sliding/global", DONE),
    Entry("gemma4-26b-a4b", "gemma4_26b_a4b", "Gemma 4 (26B-A4B)", "2026-04-02", "sparse MoE + GQA + sliding/global", DONE),
    Entry("gemma4-12b", "gemma4_12b", "Gemma 4 (12B)", "2026-06-03", "GQA + QK-Norm + sliding/global", DONE),
    Entry("llama3.2-3b", "llama3_2_3b", "Llama 3.2 (3B)", "2024-09-25", "GQA", DONE),
    Entry("qwen3-coder-flash", "qwen3_coder_flash", "Qwen3 Coder Flash (30B-A3B)", "2025-07-31", "sparse MoE + GQA", DONE),
    Entry("kimi-k2.5", "kimi_k2_5", "Kimi K2.5 (1T)", "2026-01-27", "sparse MoE + MLA", DONE),
    Entry("step-3.5-flash", "step_3_5_flash", "Step 3.5 Flash (196B)", "2026-02-01", "sparse MoE + sliding/global", DONE),
    Entry("nanbeige-4.1", "nanbeige_4_1", "Nanbeige 4.1 (3B)", "2026-02-10", "GQA", DONE),
    Entry("minimax-m2.5", "minimax_m2_5", "MiniMax-M2.5 (230B)", "2026-02-12", "sparse MoE + GQA + QK-Norm", DONE),
    Entry("tiny-aya", "tiny_aya", "Tiny Aya (3.35B)", "2026-02-13", "GQA + parallel transformer blocks", DONE),
    Entry("ling-2.5", "ling_2_5", "Ling 2.5 (1T)", "2026-02-15", "Lightning attn + MLA hybrid", DONE),
    Entry("qwen3.5", "qwen3_5", "Qwen3.5 (397B)", "2026-02-16", "Gated DeltaNet + gated attn hybrid", DONE),
    Entry("sarvam-30b", "sarvam_30b", "Sarvam (30B)", "2026-03-03", "sparse MoE + GQA + QK-Norm", DONE),
    Entry("sarvam-105b", "sarvam_105b", "Sarvam (105B)", "2026-03-03", "sparse MoE + MLA", DONE),
    Entry("phi-4", "phi_4", "Phi-4 (14B)", "2024-12-12", "GQA + RoPE", DONE),
    Entry("xlstm-7b", "xlstm_7b", "xLSTM (7B)", "2025-03-17", "recurrent mLSTM (no attention)", DONE),
    Entry("glm-4.5-air", "glm_4_5_air", "GLM-4.5-Air (106B)", "2025-07-28", "sparse MoE + GQA", DONE),
    Entry("intellect-3", "intellect_3", "INTELLECT-3 (106B)", "2025-11-26", "sparse MoE + GQA", DONE),
    Entry("longcat-flash-lite", "longcat_flash_lite", "LongCat-Flash-Lite (68.5B-A3B)", "2026-01-28", "sparse MoE + MLA", DONE),
    Entry("mistral-small-4", "mistral_small_4", "Mistral Small 4 (119B)", "2026-03-16", "sparse MoE + MLA", DONE),
    Entry("nemotron3-nano-4b", "nemotron3_nano_4b", "Nemotron 3 Nano (4B)", "2026-03-16", "dense Mamba-2 hybrid + GQA", DONE),
    Entry("minimax-m2.7", "minimax_m2_7", "MiniMax M2.7 (230B)", "2026-03-18", "sparse MoE + GQA + QK-Norm", DONE),
    Entry("gemma4-e2b", "gemma4_e2b", "Gemma 4 (E2B)", "2026-04-02", "MQA + sliding/global", DONE),
    Entry("gemma4-e4b", "gemma4_e4b", "Gemma 4 (E4B)", "2026-04-02", "GQA + sliding/global", DONE),
    Entry("deepseek-v4-flash", "deepseek_v4_flash", "DeepSeek V4-Flash (284B)", "2026-04-24", "sparse MoE + compressed sparse attn", DONE),
    Entry("deepseek-v4-pro", "deepseek_v4_pro", "DeepSeek V4-Pro (1.6T)", "2026-04-24", "sparse MoE + compressed sparse attn + mHC", DONE),
    Entry("laguna-xs.2", "laguna_xs_2", "Laguna XS.2 (33B)", "2026-04-28", "sparse MoE + gated GQA + sliding/global", DONE),
    Entry("zaya1-8b", "zaya1_8b", "ZAYA1-8B (8.4B)", "2026-05-06", "sparse MoE + CCA/GQA + top-1 routing", DONE),
    Entry("glm-5.1", "glm_5_1", "GLM-5.1 (744B)", "", "sparse MoE + MLA", DONE),
    Entry("qwen3.6-35b-a3b", "qwen3_6_35b_a3b", "Qwen3.6 (35B-A3B)", "", "sparse hybrid attention", DONE),
    Entry("kimi-k2.6", "kimi_k2_6", "Kimi K2.6 (1T)", "", "sparse MoE + MLA", DONE),
    Entry("qwen3.6-27b", "qwen3_6_27b", "Qwen3.6 (27B)", "", "dense GQA", DONE),
    Entry("mimo-v2.5", "mimo_v2_5", "Xiaomi MiMo-V2.5 (310B)", "", "sparse MoE", DONE),
    Entry("mimo-v2.5-pro", "mimo_v2_5_pro", "Xiaomi MiMo-V2.5-Pro (1.02T)", "", "sparse MoE", DONE),
    Entry("ling-2.6", "ling_2_6", "Ling 2.6 (1T)", "", "sparse hybrid linear attn", DONE),
    Entry("hunyuan-3-preview", "hunyuan_3_preview", "Tencent Hy3-preview (295B-A21B)", "", "sparse MoE", DONE),
    Entry("granite-4.1", "granite_4_1", "Granite 4.1 (30B)", "", "dense attention", DONE),
    Entry("command-a-plus", "command_a_plus", "Command A+ (218B-A25B)", "", "sparse MoE", DONE),
    Entry("lfm2.5-1.2b", "lfm2_5_1_2b", "LFM2.5 (1.2B)", "", "dense attention", DONE),
    Entry("lfm2.5-350m", "lfm2_5_350m", "LFM2.5 (350M)", "", "dense attention", DONE),
    Entry("lfm2.5-8b-a1b", "lfm2_5_8b_a1b", "LFM2.5 (8B-A1B)", "", "sparse MoE", DONE),
    Entry("mellum2-thinking", "mellum2_thinking", "JetBrains Mellum2 Thinking (12B-A2.5B)", "", "sparse MoE", DONE),
    Entry("nemotron3-ultra", "nemotron3_ultra", "Nemotron 3 Ultra (550B-A55B)", "", "sparse hybrid Mamba + MoE", DONE),
]
# fmt: on

_BY_SLUG = {e.slug: e for e in REGISTRY}


def get(slug: str) -> Entry:
    if slug not in _BY_SLUG:
        raise KeyError(f"unknown model slug: {slug!r} (try `cli list`)")
    return _BY_SLUG[slug]


def load(slug: str) -> ModuleType:
    """Import and return the model module for ``slug`` (raises if not implemented yet)."""
    entry = get(slug)
    return importlib.import_module(f"llm_gallery.models.{entry.module}")


def done_slugs() -> list[str]:
    return [e.slug for e in REGISTRY if e.status == DONE]
