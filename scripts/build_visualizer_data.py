"""Build static JSON used by the interactive model-code visualizer.

The visualizer is intentionally generated from the existing self-contained model
files. It does not ask model authors to add inline markers, and it does not import
the model modules themselves.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

import markdown
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "llm_gallery" / "models"
DATA_DIR = ROOT / "web" / "data"
ARCHITECTURE_MANIFEST_PATH = ROOT / "web" / "assets" / "architectures" / "manifest.json"
HOTSPOTS_DIR = ROOT / "web" / "assets" / "architectures" / "hotspots"
CONCEPTS_DIR = ROOT / "llm_gallery" / "concepts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_gallery.models import registry  # noqa: E402,I001


CLASS_ROLES = {
    "Config": "config",
    "LayerNorm": "norm",
    "RMSNorm": "norm",
    "GemmaRMSNorm": "norm",
    "CausalSelfAttention": "attention",
    "Attention": "attention",
    "GatedAttention": "attention",
    "MLA": "attention",
    "MiniMaxSparseAttention": "attention",
    "LinearAttention": "mixer",
    "GatedDeltaNet": "mixer",
    "Mamba": "mixer",
    "mLSTM": "mixer",
    "MLP": "mlp",
    "SwiGLU": "mlp",
    "GeGLU": "mlp",
    "MoE": "moe",
    "DeepseekMoE": "moe",
    "Expert": "expert",
    "Block": "block",
    "MambaLayer": "block",
    "AttentionLayer": "block",
    "FFNLayer": "block",
    "Model": "model",
}

FUNCTION_ROLES = {
    "precompute_rope": "position",
    "rotate_half": "position",
    "apply_rope": "position",
    "repeat_kv": "attention_helper",
}

COMPUTE_ROLES = {"attention", "mixer", "mlp", "moe"}
CORE_ROLE_ORDER = {
    "config": 0,
    "presets": 1,
    "norm": 2,
    "position": 3,
    "attention": 4,
    "mixer": 5,
    "attention_helper": 6,
    "mlp": 7,
    "expert": 8,
    "moe": 9,
    "block": 10,
    "model": 11,
    "helper": 12,
}

ROLE_LABELS = {
    "config": "Config",
    "presets": "Presets",
    "norm": "Normalization",
    "position": "Position Encoding",
    "attention": "Attention",
    "mixer": "Sequence Mixer",
    "attention_helper": "Attention Helper",
    "mlp": "Feed Forward",
    "expert": "Expert",
    "moe": "Mixture of Experts",
    "block": "Block",
    "model": "Full Model",
    "helper": "Helper",
}

# Concept mapping: every section and anchor gets a concept_id pointing into
# llm_gallery/concepts/*.md (bundled as web/data/concepts.json). Resolution order for
# sections: label match, then attention-variant detection via anchors, then role.
SECTION_ROLE_CONCEPTS = {
    "config": "hyperparameters",
    "presets": "hyperparameters",
    "position": "rope",
    "attention": "attention",
    "attention_helper": "gqa",
    "mlp": "mlp",
    "expert": "moe",
    "moe": "moe",
    "block": "transformer-block",
    "model": "decoder-model",
}

SECTION_LABEL_CONCEPTS = {
    "LayerNorm": "layernorm",
    "RMSNorm": "rmsnorm",
    "GemmaRMSNorm": "rmsnorm",
    "MLA": "mla",
    "MiniMaxSparseAttention": "sparse-attention",
    "GatedAttention": "gated-attention",
    "SwiGLU": "glu-feedforward",
    "GeGLU": "glu-feedforward",
    "DenseMLP": "glu-feedforward",
    "Mamba": "mamba",
    "LinearAttention": "linear-attention",
    "GatedDeltaNet": "linear-attention",
    "mLSTM": "mlstm",
}

ANCHOR_ROLE_CONCEPTS = {
    "embedding": "token-embeddings",
    "position_embedding": "learned-positions",
    "embedding_dropout": "dropout",
    "attention_dropout": "dropout",
    "mlp_dropout": "dropout",
    "qkv": "qkv-projections",
    "output_projection": "qkv-projections",
    "qk_norm": "qk-norm",
    "rope": "rope",
    "kv_share": "gqa",
    "attention_math": "attention",
    "sparse_attention": "sparse-attention",
    "latent_query": "mla",
    "latent_kv": "mla",
    "activation": "activations",
    "residual_attn": "residuals",
    "residual_mlp": "residuals",
    "block_stack": "transformer-block",
    "lm_head": "lm-head",
    "output_head": "lm-head",
    "router": "moe-router",
    "topk": "moe-router",
    "experts": "moe",
    "shared_expert": "shared-experts",
    "parallel": "parallel-block",
}

ANCHOR_ROLE_ORDER = {
    "embedding": 0,
    "position_embedding": 1,
    "embedding_dropout": 2,
    "norm_1": 2,
    "norm_2": 3,
    "block_stack": 1,
    "final_norm": 2,
    "lm_head": 3,
    "output_head": 4,
    "residual_attn": 5,
    "residual_mlp": 6,
    "parallel": 7,
    "qkv": 8,
    "qk_norm": 9,
    "rope": 10,
    "kv_share": 11,
    "attention_math": 12,
    "output_projection": 13,
    "attention_dropout": 14,
    "sparse_attention": 15,
    "latent_query": 16,
    "latent_kv": 17,
    "mlp_gate": 18,
    "activation": 19,
    "mlp_output": 20,
    "mlp_dropout": 21,
    "router": 22,
    "topk": 23,
    "experts": 24,
    "shared_expert": 25,
    "mixer_state": 26,
}

GALLERY_SOURCE_KEYS = {
    "gpt2-xl": "gpt-2-xl-1-5b",
    "llama3-8b": "llama-3-8b",
    "llama3.2-1b": "llama-3-2-1b",
    "olmo2-7b": "olmo-2-7b",
    "deepseek-v3": "deepseek-v3",
    "deepseek-r1": "deepseek-r1",
    "gemma3-27b": "gemma-3-27b",
    "mistral-small-3.1": "mistral-small-3-1-24b",
    "llama4-maverick": "llama-4-maverick",
    "qwen3-0.6b": "qwen3-0-6b",
    "qwen3-235b-a22b": "qwen3-235b-a22b",
    "qwen3-30b-a3b": "qwen3-30b-a3b",
    "qwen3-32b": "qwen3-32b",
    "qwen3-4b": "qwen3-4b",
    "qwen3-8b": "qwen3-8b",
    "smollm3-3b": "smollm3-3b",
    "kimi-k2": "kimi-k2",
    "glm-4.5": "glm-4-5-355b",
    "gpt-oss-120b": "gpt-oss-120b",
    "gpt-oss-20b": "gpt-oss-20b",
    "gemma3-270m": "gemma-3-270m",
    "grok-2.5": "grok-2-5-270b",
    "qwen3-next-80b-a3b": "qwen3-next-80b-a3b",
    "minimax-m2": "minimax-m2-230b",
    "kimi-linear": "kimi-linear-48b-a3b",
    "olmo3-32b": "olmo-3-32b",
    "olmo3-7b": "olmo-3-7b",
    "deepseek-v3.2": "deepseek-v3-2",
    "mistral-large-3": "mistral-large-3",
    "nemotron3-nano-30b": "nemotron-3-nano-30b-a3b",
    "mimo-v2-flash": "xiaomi-mimo-v2-flash-309b",
    "glm-4.7": "glm-4-7-355b",
    "arcee-trinity-large": "arcee-ai-trinity-large-400b",
    "glm-5": "glm-5-744b",
    "nemotron3-super-120b": "nemotron-3-super-120b-a12b",
    "gemma4-31b": "gemma-4-31b",
    "gemma4-26b-a4b": "gemma-4-26b-a4b",
    "gemma4-12b": "gemma-4-12b",
    "llama3.2-3b": "llama-3-2-3b",
    "qwen3-coder-flash": "qwen3-coder-flash-30b-a3b",
    "kimi-k2.5": "kimi-k2-5",
    "step-3.5-flash": "step-3-5-flash-196b",
    "nanbeige-4.1": "nanbeige-4-1-3b",
    "minimax-m2.5": "minimax-m2-5-230b",
    "tiny-aya": "tiny-aya-3-35b",
    "ling-2.5": "ling-2-5-1t",
    "qwen3.5": "qwen3-5-397b",
    "sarvam-30b": "sarvam-30b",
    "sarvam-105b": "sarvam-105b",
    "phi-4": "phi-4",
    "xlstm-7b": "xlstm-7b",
    "glm-4.5-air": "glm-4-5-air",
    "intellect-3": "intellect-3",
    "longcat-flash-lite": "longcat-flash-lite-68-5b-a3b",
    "mistral-small-4": "mistral-small-4",
    "nemotron3-nano-4b": "nemotron-3-nano-4b",
    "minimax-m2.7": "minimax-m2-7-230b",
    "gemma4-e2b": "gemma-4-e2b",
    "gemma4-e4b": "gemma-4-e4b",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "laguna-xs.2": "laguna-xs-2",
    "zaya1-8b": "zaya1-8b",
    "glm-5.1": "glm-5-1",
    "qwen3.6-35b-a3b": "qwen3-6-35b-a3b",
    "kimi-k2.6": "kimi-k2-6",
    "qwen3.6-27b": "qwen3-6-27b",
    "mimo-v2.5": "xiaomi-mimo-v2-5-310b",
    "mimo-v2.5-pro": "xiaomi-mimo-v2-5-pro-1-02t",
    "ling-2.6": "ling-2-6-1t",
    "hunyuan-3-preview": "tencent-hy3-preview-295b-a21b",
    "granite-4.1": "granite-4-1-30b",
    "command-a-plus": "command-a-218b-a25b",
    "lfm2.5-1.2b": "lfm2-5-1-2b",
    "lfm2.5-350m": "lfm2-5-350m",
    "lfm2.5-8b-a1b": "lfm2-5-8b-a1b",
    "mellum2-thinking": "jetbrains-mellum2-thinking-12b-a2-5b",
    "nemotron3-ultra": "nemotron-3-ultra-550b-a55b",
    "north-mini-code": "north-mini-code-30b-a3b",
    "kimi-k2.7-code": "kimi-k2-7-code",
    "minimax-m3": "minimax-m3-428b",
    "vibethinker-3b": "vibethinker-3b",
    "glm-5.2": "glm-5-2",
}

GALLERY_BASE_URL = "https://sebastianraschka.com/llm-architecture-gallery/"

GALLERY_ARTICLE_URLS = {
    "gpt2-xl": "https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the#%C2%A72-coming-from-gpt-2",
    "llama3-8b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A723-olmo-2-summary",
    "llama3.2-1b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A761-qwen3-dense",
    "olmo2-7b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A723-olmo-2-summary",
    "deepseek-v3": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A75-llama-4",
    "deepseek-r1": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A78-kimi-k2-and-kimi-k2-thinking",
    "gemma3-27b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A74-mistral-small-31",
    "mistral-small-3.1": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A74-mistral-small-31",
    "llama4-maverick": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A75-llama-4",
    "qwen3-0.6b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A761-qwen3-dense",
    "qwen3-235b-a22b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A762-qwen3-moe",
    "qwen3-30b-a3b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A79-gpt-oss",
    "qwen3-32b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A715-olmo-3-thinking",
    "qwen3-4b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A77-smollm3",
    "qwen3-8b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A715-olmo-3-thinking",
    "smollm3-3b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A77-smollm3",
    "kimi-k2": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A78-kimi-k2-and-kimi-k2-thinking",
    "glm-4.5": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A711-glm-45",
    "gpt-oss-120b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A79-gpt-oss",
    "gpt-oss-20b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A79-gpt-oss",
    "gemma3-270m": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A74-mistral-small-31",
    "grok-2.5": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A710-grok-25",
    "qwen3-next-80b-a3b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A7121-expert-size-and-number",
    "minimax-m2": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A7131-per-layer-qk-norm",
    "kimi-linear": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A7144-kimi-linear-vs-qwen3-next",
    "olmo3-32b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A715-olmo-3-thinking",
    "olmo3-7b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A715-olmo-3-thinking",
    "deepseek-v3.2": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A716-deepseek-v32",
    "mistral-large-3": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A717-mistral-3-large",
    "nemotron3-nano-30b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A7181-nemotron-3-nano",
    "mimo-v2-flash": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A719-xiaomi-mimo-v2-flash",
    "glm-4.7": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A721-glm-5",
    "arcee-trinity-large": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A720-arcee-ai-trinity-large",
    "glm-5": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A721-glm-5",
    "nemotron3-super-120b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A7182-nemotron-3-super",
    "gemma4-31b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A723-gemma-4",
    "gemma4-26b-a4b": "https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison#%C2%A723-gemma-4",
    "llama3.2-3b": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A77-nanbeige-41-3b-a-strong-llama-3-successor",
    "qwen3-coder-flash": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A74-qwen3-coder-next-an-attention-hybrid-for-coding",
    "kimi-k2.5": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A72-moonshot-ais-kimi-k25-a-deepseek-like-model-at-a-1-trillion-parameter-scale",
    "step-3.5-flash": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A73-stepfuns-step-35-flash-good-performance-at-great-tokens-sec-throughput",
    "nanbeige-4.1": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A77-nanbeige-41-3b-a-strong-llama-3-successor",
    "minimax-m2.5": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A76-minimax-m25-a-strong-coder-with-only-230b-parameters",
    "tiny-aya": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A710-tiny-aya-a-335b-model-with-strong-multilingual-support",
    "ling-2.5": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A79-ant-groups-ling-25-1t-with-lightning-attention",
    "qwen3.5": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A78-qwen35-and-the-continuation-of-hybrid-attention",
    "sarvam-30b": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A7update-1-sarvam-30b-and-105b-mar-6-2026",
    "sarvam-105b": "https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight#%C2%A7update-1-sarvam-30b-and-105b-mar-6-2026",
    "gemma4-e2b": "https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures#%C2%A71-reusing-kv-tensors-across-layers-to-shrink-the-cache-gemma-4",
    "gemma4-e4b": "https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures#%C2%A71-reusing-kv-tensors-across-layers-to-shrink-the-cache-gemma-4",
    "deepseek-v4-flash": "https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures#%C2%A75-csahca-mhc-and-compressed-attention-caches-deepseek-v4",
    "deepseek-v4-pro": "https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures#%C2%A75-csahca-mhc-and-compressed-attention-caches-deepseek-v4",
    "laguna-xs.2": "https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures#%C2%A73-layer-wise-attention-budgeting-laguna-xs2",
    "zaya1-8b": "https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures#%C2%A74-compressed-convolutional-attention-zaya1-8b",
}

DIAGRAM_PALETTES = {
    "gpt2-xl": {"accent": "#3080a8", "accentFill": "#98b8c8"},
    "llama3-8b": {"accent": "#0070b0", "accentFill": "#c0e0e8"},
    "llama3.2-1b": {"accent": "#d01070", "accentFill": "#d8b0c0"},
    "olmo2-7b": {"accent": "#f05098", "accentFill": "#e0b0c8"},
    "deepseek-v3": {"accent": "#f86050", "accentFill": "#f09088"},
    "deepseek-r1": {"accent": "#f86050", "accentFill": "#f09088"},
    "gemma3-27b": {"accent": "#f07000", "accentFill": "#e0d0a8"},
    "mistral-small-3.1": {"accent": "#f86050", "accentFill": "#e89888"},
    "llama4-maverick": {"accent": "#00a0f8", "accentFill": "#90c0e0"},
    "qwen3-0.6b": {"accent": "#58c0ff", "accentFill": "#60c0f8"},
    "qwen3-235b-a22b": {"accent": "#00a0f8", "accentFill": "#90c0e0"},
    "qwen3-30b-a3b": {"accent": "#00a0f8", "accentFill": "#90c0e0"},
    "qwen3-32b": {"accent": "#58c0ff", "accentFill": "#60c0f8"},
    "qwen3-4b": {"accent": "#58c0ff", "accentFill": "#60c0f8"},
    "qwen3-8b": {"accent": "#58c0ff", "accentFill": "#58b0e0"},
    "smollm3-3b": {"accent": "#f8b000", "accentFill": "#e8d888"},
    "kimi-k2": {"accent": "#60d838", "accentFill": "#68d040"},
    "glm-4.5": {"accent": "#00a888", "accentFill": "#78c0b8"},
    "gpt-oss-120b": {"accent": "#8840e0", "accentFill": "#b090e0"},
    "gpt-oss-20b": {"accent": "#4878e0", "accentFill": "#88a0e0"},
    "gemma3-270m": {"accent": "#f07000", "accentFill": "#e0d0a8"},
    "grok-2.5": {"accent": "#52b9ee", "accentFill": "#52b9ee"},
    "qwen3-next-80b-a3b": {"accent": "#38b0c8", "accentFill": "#78c0d0"},
    "minimax-m2": {"accent": "#e84060", "accentFill": "#e090a0"},
    "kimi-linear": {"accent": "#00a0f8", "accentFill": "#407090"},
    "olmo3-32b": {"accent": "#f05098", "accentFill": "#e0b0c8"},
    "olmo3-7b": {"accent": "#f05098", "accentFill": "#e0b0c8"},
    "deepseek-v3.2": {"accent": "#2090f8", "accentFill": "#80b0e8"},
    "mistral-large-3": {"accent": "#e0a068", "accentFill": "#e0a068"},
    "nemotron3-nano-30b": {"accent": "#70b000", "accentFill": "#b0c888"},
    "mimo-v2-flash": {"accent": "#e89870", "accentFill": "#e89870"},
    "glm-4.7": {"accent": "#00a888", "accentFill": "#78c0b8"},
    "arcee-trinity-large": {"accent": "#400090", "accentFill": "#a898c0"},
    "glm-5": {"accent": "#60a8f0", "accentFill": "#60a8f0"},
    "nemotron3-super-120b": {"accent": "#70b000", "accentFill": "#b0c888"},
    "gemma4-31b": {"accent": "#3880f8", "accentFill": "#58a0f8"},
    "gemma4-26b-a4b": {"accent": "#3880f8", "accentFill": "#5090e0"},
    "gemma4-12b": {"accent": "#3880f8", "accentFill": "#4880f0"},
    "llama3.2-3b": {"accent": "#d01070", "accentFill": "#d8b0c0"},
    "qwen3-coder-flash": {"accent": "#00a0f8", "accentFill": "#90c0e0"},
    "kimi-k2.5": {"accent": "#2040b0", "accentFill": "#8088c0"},
    "step-3.5-flash": {"accent": "#0070b0", "accentFill": "#78a8c8"},
    "nanbeige-4.1": {"accent": "#007800", "accentFill": "#a8d090"},
    "minimax-m2.5": {"accent": "#e84060", "accentFill": "#e090a0"},
    "tiny-aya": {"accent": "#58c088", "accentFill": "#58c088"},
    "ling-2.5": {"accent": "#f07000", "accentFill": "#e0b090"},
    "qwen3.5": {"accent": "#0070b0", "accentFill": "#78a8c8"},
    "sarvam-30b": {"accent": "#f89030", "accentFill": "#e8b090"},
    "sarvam-105b": {"accent": "#f89030", "accentFill": "#e8b8a0"},
    "phi-4": {"accent": "#68a0f0", "accentFill": "#68a0f0"},
    "xlstm-7b": {"accent": "#e8c850", "accentFill": "#e8c850"},
    "glm-4.5-air": {"accent": "#00a888", "accentFill": "#78c0b8"},
    "intellect-3": {"accent": "#52b9ee", "accentFill": "#52b9ee"},
    "longcat-flash-lite": {"accent": "#3878f0", "accentFill": "#4078f0"},
    "mistral-small-4": {"accent": "#e0a068", "accentFill": "#e0a068"},
    "nemotron3-nano-4b": {"accent": "#70b000", "accentFill": "#b0c888"},
    "minimax-m2.7": {"accent": "#e88078", "accentFill": "#e88078"},
    "gemma4-e2b": {"accent": "#3880f8", "accentFill": "#4080f0"},
    "gemma4-e4b": {"accent": "#3880f8", "accentFill": "#4080f0"},
    "deepseek-v4-flash": {"accent": "#2090f8", "accentFill": "#70a0d0"},
    "deepseek-v4-pro": {"accent": "#2090f8", "accentFill": "#70a0d0"},
    "laguna-xs.2": {"accent": "#50b000", "accentFill": "#a8d870"},
    "zaya1-8b": {"accent": "#d8a850", "accentFill": "#d8a850"},
    "glm-5.1": {"accent": "#3090e0", "accentFill": "#48b0f0"},
    "qwen3.6-35b-a3b": {"accent": "#0070b0", "accentFill": "#a8c0e0"},
    "kimi-k2.6": {"accent": "#2040b0", "accentFill": "#b0a8d0"},
    "qwen3.6-27b": {"accent": "#0070b0", "accentFill": "#78a8d8"},
    "mimo-v2.5": {"accent": "#e89058", "accentFill": "#e89058"},
    "mimo-v2.5-pro": {"accent": "#e09870", "accentFill": "#e09870"},
    "ling-2.6": {"accent": "#f0b860", "accentFill": "#f0b860"},
    "hunyuan-3-preview": {"accent": "#70d0f8", "accentFill": "#a0e0f0"},
    "granite-4.1": {"accent": "#7090e8", "accentFill": "#7090e8"},
    "command-a-plus": {"accent": "#58c088", "accentFill": "#58c088"},
    "lfm2.5-1.2b": {"accent": "#80b030", "accentFill": "#80b038"},
    "lfm2.5-350m": {"accent": "#80b030", "accentFill": "#80b038"},
    "lfm2.5-8b-a1b": {"accent": "#80b028", "accentFill": "#90c038"},
    "mellum2-thinking": {"accent": "#e85090", "accentFill": "#e85090"},
    "nemotron3-ultra": {"accent": "#70b000", "accentFill": "#b0c888"},
    "north-mini-code": {"accent": "#58c088", "accentFill": "#58c088"},
    "kimi-k2.7-code": {"accent": "#2040b0", "accentFill": "#b0a8d0"},
    "minimax-m3": {"accent": "#e84060", "accentFill": "#e090a0"},
    "vibethinker-3b": {"accent": "#58c0ff", "accentFill": "#60c0f8"},
    "glm-5.2": {"accent": "#60a8f0", "accentFill": "#60a8f0"},
}


def slugify(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "-", value).replace("_", "-").lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "section"


def assignment_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]

    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Tuple):
            names.update(elt.id for elt in target.elts if isinstance(elt, ast.Name))
    return names


def literal_constants(tree: ast.Module) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for node in tree.body:
        names = assignment_names(node)
        if not names:
            continue
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = getattr(node, "value", None)
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            continue
        for name in names:
            if name.isupper():
                constants[name] = literal
    return constants


def unique_id(base: str, used: dict[str, int]) -> str:
    count = used.get(base, 0)
    used[base] = count + 1
    if count == 0:
        return base
    return f"{base}-{count + 1}"


def section_from_node(node: ast.AST, source_lines: list[str], used: dict[str, int]) -> dict[str, Any] | None:
    if isinstance(node, ast.ClassDef):
        role = CLASS_ROLES.get(node.name, "helper")
        label = node.name
        symbol_type = "class"
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        role = FUNCTION_ROLES.get(node.name, "helper")
        label = node.name
        symbol_type = "function"
    else:
        return None

    line_start = int(node.lineno)
    line_end = int(getattr(node, "end_lineno", node.lineno))
    section_id = unique_id(role if role in CORE_ROLE_ORDER else slugify(label), used)
    doc = ast.get_docstring(node, clean=True)
    return {
        "id": section_id,
        "role": role,
        "label": label,
        "symbol_type": symbol_type,
        "line_start": line_start,
        "line_end": line_end,
        "summary": doc.splitlines()[0] if doc else "",
        "line_count": line_end - line_start + 1,
        "source_preview": source_lines[line_start - 1].strip(),
    }


def presets_section(tree: ast.Module, used: dict[str, int]) -> dict[str, Any] | None:
    start = end = None
    for node in tree.body:
        names = assignment_names(node)
        if "PRESETS" in names:
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", node.lineno))
        elif start is not None and "DEFAULT_PRESET" in names:
            end = int(getattr(node, "end_lineno", node.lineno))
            break
    if start is None or end is None:
        return None
    return {
        "id": unique_id("presets", used),
        "role": "presets",
        "label": "PRESETS",
        "symbol_type": "assignment",
        "line_start": start,
        "line_end": end,
        "summary": "Runnable and reference model configurations.",
        "line_count": end - start + 1,
        "source_preview": "PRESETS",
    }


def extract_sections(tree: ast.Module, source_lines: list[str]) -> list[dict[str, Any]]:
    used: dict[str, int] = {}
    sections: list[dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            section = section_from_node(node, source_lines, used)
            if section is not None:
                sections.append(section)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and "PRESETS" in assignment_names(node):
            section = presets_section(tree, used)
            if section is not None:
                sections.append(section)

    sections.sort(key=lambda item: (item["line_start"], item["line_end"], item["id"]))
    return sections


def first_section(sections: list[dict[str, Any]], *roles: str) -> dict[str, Any] | None:
    role_set = set(roles)
    return next((section for section in sections if section["role"] in role_set), None)


def all_sections(sections: list[dict[str, Any]], *roles: str) -> list[dict[str, Any]]:
    role_set = set(roles)
    return [section for section in sections if section["role"] in role_set]


def section_for_symbol(sections: list[dict[str, Any]], label: str, symbol_type: str = "class") -> dict[str, Any] | None:
    return next(
        (
            section
            for section in sections
            if section["label"] == label and section["symbol_type"] == symbol_type
        ),
        None,
    )


def class_defs(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def method_named(class_node: ast.ClassDef, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ),
        None,
    )


def node_range(node: ast.AST) -> tuple[int, int]:
    return int(node.lineno), int(getattr(node, "end_lineno", node.lineno))


def statement_text(node: ast.AST, source_lines: list[str]) -> str:
    start, end = node_range(node)
    return "\n".join(source_lines[start - 1 : end])


def method_statements(method: ast.FunctionDef | ast.AsyncFunctionDef | None) -> list[ast.stmt]:
    if method is None:
        return []
    statements = [node for node in ast.walk(method) if isinstance(node, ast.stmt) and node is not method]
    return sorted(statements, key=lambda node: (int(node.lineno), int(getattr(node, "end_lineno", node.lineno))))


def matching_statements(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    source_lines: list[str],
    *,
    all_of: tuple[str, ...] = (),
    any_of: tuple[str, ...] = (),
) -> list[ast.stmt]:
    matches: list[ast.stmt] = []
    for statement in method_statements(method):
        text = statement_text(statement, source_lines)
        if all(needle in text for needle in all_of) and (
            not any_of or any(needle in text for needle in any_of)
        ):
            matches.append(statement)
    return matches


def combined_range(statements: list[ast.stmt]) -> tuple[int, int] | None:
    if not statements:
        return None
    return (
        min(int(statement.lineno) for statement in statements),
        max(int(getattr(statement, "end_lineno", statement.lineno)) for statement in statements),
    )


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def assigned_self_attrs(statements: list[ast.stmt]) -> set[str]:
    attrs: set[str] = set()
    for statement in statements:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                attrs.add(target.attr)
    return attrs


def nested_block_nodes(statement: ast.stmt) -> set[ast.AST]:
    nested: set[ast.AST] = set()
    for field in ("body", "orelse", "finalbody", "handlers"):
        for child in getattr(statement, field, None) or []:
            nested.update(ast.walk(child))
    return nested


def self_attr_usage_ranges(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    attr_names: set[str],
) -> list[tuple[int, int]]:
    if method is None or not attr_names:
        return []
    ranges: list[tuple[int, int]] = []
    for statement in method_statements(method):
        nested = nested_block_nodes(statement)
        hits = [
            node
            for node in ast.walk(statement)
            if node not in nested
            and isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in attr_names
        ]
        if not hits:
            continue
        if getattr(statement, "body", None):
            # Compound statement (for/if/while/with): highlight only the header
            # references, not the whole nested body.
            ranges.extend(node_range(node) for node in hits)
        else:
            ranges.append(node_range(statement))
    return merge_ranges(ranges)


def add_anchor(
    anchors: list[dict[str, Any]],
    used: dict[str, int],
    section: dict[str, Any] | None,
    source_lines: list[str],
    role: str,
    label: str,
    line_range: tuple[int, int] | None,
    usage_ranges: list[tuple[int, int]] | None = None,
) -> dict[str, Any] | None:
    if section is None or line_range is None:
        return None
    line_start, line_end = line_range
    if line_start < section["line_start"] or line_end > section["line_end"]:
        return None
    ranges = [{"kind": "definition", "line_start": line_start, "line_end": line_end}]
    for usage_start, usage_end in usage_ranges or []:
        if usage_start < section["line_start"] or usage_end > section["line_end"]:
            continue
        if usage_start <= line_end and usage_end >= line_start:
            continue
        ranges.append({"kind": "usage", "line_start": usage_start, "line_end": usage_end})
    ranges.sort(key=lambda item: (item["line_start"], item["line_end"]))
    anchor_id = unique_id(f"{section['id']}.{slugify(role)}", used)
    anchor = {
        "id": anchor_id,
        "section_id": section["id"],
        "role": role,
        "label": label,
        "line_start": line_start,
        "line_end": line_end,
        "line_count": line_end - line_start + 1,
        "ranges": ranges,
        "source_preview": source_lines[line_start - 1].strip(),
    }
    anchors.append(anchor)
    return anchor


def add_anchor_from_matches(
    anchors: list[dict[str, Any]],
    used: dict[str, int],
    section: dict[str, Any] | None,
    source_lines: list[str],
    role: str,
    label: str,
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    *,
    all_of: tuple[str, ...] = (),
    any_of: tuple[str, ...] = (),
    usage_method: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
) -> dict[str, Any] | None:
    statements = matching_statements(method, source_lines, all_of=all_of, any_of=any_of)
    usage_ranges = self_attr_usage_ranges(usage_method, assigned_self_attrs(statements))
    return add_anchor(
        anchors,
        used,
        section,
        source_lines,
        role,
        label,
        combined_range(statements),
        usage_ranges=usage_ranges,
    )


def extract_config_defaults(tree: ast.Module) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    config = class_defs(tree).get("Config")
    if config is None:
        return defaults
    for statement in config.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        if statement.value is None:
            continue
        try:
            defaults[statement.target.id] = ast.literal_eval(statement.value)
        except (ValueError, SyntaxError):
            continue
    return defaults


def extract_marked_notes(tree: ast.Module) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    marker_re = re.compile(r"\b(ASSUMPTION|NOTE):\s*(.+?)(?=(?:\s+\b(?:ASSUMPTION|NOTE):)|$)")

    def collect(doc: str | None) -> None:
        if not doc:
            return
        text = " ".join(line.strip() for line in doc.splitlines() if line.strip())
        for match in marker_re.finditer(text):
            kind = match.group(1).lower()
            body = match.group(2).strip()
            if not body:
                continue
            key = (kind, body)
            if key not in seen:
                seen.add(key)
                notes.append({"kind": kind, "text": body})

    collect(ast.get_docstring(tree, clean=True))
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            collect(ast.get_docstring(node, clean=True))
    return notes


def pygments_class(ttype: Any, text: str) -> str | None:
    if ttype in Token.Keyword:
        return "py-keyword"
    if ttype in Token.Name.Decorator:
        return "py-decorator"
    if ttype in Token.Literal.String:
        return "py-string"
    if ttype in Token.Comment:
        return "py-comment"
    if ttype in Token.Literal.Number:
        return "py-number"
    if ttype in Token.Name.Function:
        return "py-function"
    if ttype in Token.Name.Class:
        return "py-class"
    if ttype in Token.Name.Builtin or text in {"self", "cls"}:
        return "py-self" if text in {"self", "cls"} else "py-builtin"
    if ttype in Token.Name.Namespace:
        return "py-module"
    if ttype in Token.Operator or ttype in Token.Punctuation:
        return "py-operator"
    return None


def highlighted_source_lines(source: str) -> list[list[dict[str, str]]]:
    highlighted: list[list[dict[str, str]]] = [[]]
    lexer = PythonLexer(stripnl=False)
    for ttype, text in lex(source, lexer):
        class_name = pygments_class(ttype, text)
        for part in text.splitlines(keepends=True):
            token_text = part[:-1] if part.endswith("\n") else part
            if token_text:
                token = {"t": token_text}
                if class_name is not None:
                    token["c"] = class_name
                highlighted[-1].append(token)
            if part.endswith("\n"):
                highlighted.append([])

    source_line_count = len(source.splitlines())
    if len(highlighted) > source_line_count and not highlighted[-1]:
        highlighted.pop()
    while len(highlighted) < source_line_count:
        highlighted.append([])
    return highlighted


def extract_anchors(tree: ast.Module, source_lines: list[str], sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    classes = class_defs(tree)

    model_node = classes.get("Model")
    model_section = section_for_symbol(sections, "Model")
    if model_node is not None:
        model_init = method_named(model_node, "__init__")
        model_forward = method_named(model_node, "forward")
        add_anchor_from_matches(
            anchors, used, model_section, source_lines, "embedding", "Token embedding", model_init,
            any_of=("self.tok_emb =", "self.wte ="),
            usage_method=model_forward,
        )
        add_anchor_from_matches(
            anchors, used, model_section, source_lines, "position_embedding", "Positional embedding", model_init,
            any_of=("self.wpe =", "pos_emb"),
            usage_method=model_forward,
        )
        add_anchor_from_matches(
            anchors, used, model_section, source_lines, "embedding_dropout", "Embedding dropout", model_init,
            any_of=("self.drop =",),
            usage_method=model_forward,
        )
        add_anchor_from_matches(
            anchors, used, model_section, source_lines, "block_stack", "Repeated blocks", model_init,
            any_of=("self.blocks", "ModuleList"),
            usage_method=model_forward,
        )
        add_anchor_from_matches(
            anchors, used, model_section, source_lines, "output_head", "Final norm + LM head", model_init,
            any_of=("self.lm_head", "self.norm", "self.ln_f"),
            usage_method=model_forward,
        )
        add_anchor_from_matches(
            anchors, used, model_section, source_lines, "final_norm", "Final normalization", model_init,
            any_of=("self.norm =", "self.ln_f ="),
            usage_method=model_forward,
        )
        add_anchor_from_matches(
            anchors, used, model_section, source_lines, "lm_head", "Linear output layer", model_init,
            any_of=("self.lm_head =", "self.output =", "self.head ="),
            usage_method=model_forward,
        )
        add_anchor_from_matches(
            anchors, used, model_section, source_lines, "rope", "RoPE cache", model_init,
            any_of=("precompute_rope", "rope_cos", "rope_sin"),
            usage_method=model_forward,
        )
        add_anchor_from_matches(
            anchors, used, model_section, source_lines, "block_stack", "Run blocks", model_forward,
            any_of=("for block in self.blocks",),
        )

    block_node = classes.get("Block")
    block_section = section_for_symbol(sections, "Block")
    if block_node is not None:
        block_init = method_named(block_node, "__init__")
        block_forward = method_named(block_node, "forward")
        add_anchor_from_matches(
            anchors, used, block_section, source_lines, "norm_1", "Attention norm", block_init,
            any_of=(
                "self.ln_1 =",
                "self.attn_norm =",
                "self.mix_norm =",
                "self.input_layernorm =",
                "self.attention_norm =",
                "self.norm1 =",
            ),
            usage_method=block_forward,
        )
        add_anchor_from_matches(
            anchors, used, block_section, source_lines, "norm_2", "Feed-forward norm", block_init,
            any_of=(
                "self.ln_2 =",
                "self.ffn_norm =",
                "self.post_attention_layernorm =",
                "self.mlp_norm =",
                "self.norm2 =",
            ),
            usage_method=block_forward,
        )
        add_anchor_from_matches(
            anchors, used, block_section, source_lines, "parallel", "Parallel attention + FFN", block_forward,
            all_of=("self.attn", "self.ffn"),
        )
        add_anchor_from_matches(
            anchors, used, block_section, source_lines, "residual_attn", "Attention residual", block_forward,
            any_of=("self.attn", "self.mix"),
        )
        add_anchor_from_matches(
            anchors, used, block_section, source_lines, "residual_mlp", "Feed-forward residual", block_forward,
            any_of=("self.ffn", "self.moe", "self.mlp"),
        )

    for class_name, class_node in classes.items():
        section = section_for_symbol(sections, class_name)
        if section is None:
            continue
        init = method_named(class_node, "__init__")
        forward = method_named(class_node, "forward")
        role = section["role"]
        if role in {"attention", "mixer"}:
            add_anchor_from_matches(
                anchors, used, section, source_lines, "qkv", "Q/K/V projections", init,
                any_of=("q_proj", "k_proj", "v_proj", "kv_a_proj", "q_a_proj", "c_attn"),
                usage_method=forward,
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "qk_norm", "QK normalization", init,
                any_of=("q_norm", "k_norm", "q_a_norm", "kv_a_norm"),
                usage_method=forward,
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "rope", "Rotary position mix", forward,
                any_of=("apply_rope", "rope"),
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "kv_share", "KV sharing / cache reduction", forward,
                any_of=("repeat_kv", "kv_latent", "k_rope"),
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "attention_math", "Attention weights", forward,
                any_of=("softmax", "@ k.transpose", "masked_fill"),
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "output_projection", "Output projection", init,
                any_of=("o_proj", "out_proj", "c_proj"),
                usage_method=forward,
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "attention_dropout", "Attention dropout", init,
                any_of=("attn_dropout", "resid_dropout"),
                usage_method=forward,
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "sparse_attention", "Sparse attention metadata", init,
                any_of=("sparse_attention", "sparse_topk", "index_topk", "IndexShare", "DSA"),
                usage_method=forward,
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "latent_query", "Low-rank query path", init,
                any_of=("q_a_proj", "q_b_proj"),
                usage_method=forward,
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "latent_kv", "Latent KV path", init,
                any_of=("kv_a_proj", "kv_b_proj", "kv_lora"),
                usage_method=forward,
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "mixer_state", "Mixer state update", forward,
                any_of=("state", "memory", "for i in range", "for s in range", "s ="),
            )
        if role == "block":
            add_anchor_from_matches(
                anchors, used, section, source_lines, "norm_1", "Layer norm", init,
                any_of=("self.norm =", "self.attn_norm =", "self.mix_norm ="),
                usage_method=forward,
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "residual_attn", "Mixer / attention residual", forward,
                any_of=("self.attn", "self.mix"),
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "residual_mlp", "Feed-forward residual", forward,
                any_of=("self.ffn",),
            )
        if role in {"mlp", "expert"}:
            add_anchor_from_matches(
                anchors, used, section, source_lines, "mlp_gate", "Gate/up projection", init,
                any_of=("gate_proj", "up_proj", "c_fc"),
                usage_method=forward,
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "mlp_output", "Activation + down projection", forward,
                any_of=("silu", "gelu", "relu", "sigmoid", "down_proj", "c_proj"),
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "activation", "Activation", forward,
                any_of=("silu", "gelu", "GELU", "relu", "sigmoid", "act("),
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "mlp_dropout", "MLP dropout", init,
                any_of=("self.dropout", "Dropout"),
                usage_method=forward,
            )
        if role == "moe":
            add_anchor_from_matches(
                anchors, used, section, source_lines, "router", "Router / gate", init,
                any_of=("self.gate", "router"),
                usage_method=forward,
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "topk", "Top-k routing", forward,
                any_of=("topk", "top_k"),
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "experts", "Expert dispatch", forward,
                any_of=("for e in range", "experts"),
            )
            add_anchor_from_matches(
                anchors, used, section, source_lines, "shared_expert", "Shared expert", init,
                any_of=("shared", "n_shared_experts"),
                usage_method=forward,
            )

    anchors.sort(
        key=lambda item: (
            item["line_start"],
            item["line_end"],
            ANCHOR_ROLE_ORDER.get(item["role"], 99),
            item["id"],
        )
    )
    return anchors


def infer_template(sections: list[dict[str, Any]], archetype: str) -> str:
    roles = {section["role"] for section in sections}
    arch = archetype.lower()
    if "parallel" in arch:
        return "parallel"
    if "mixer" in roles or "mamba" in arch or "linear attn" in arch or "deltanet" in arch or "xlstm" in arch:
        return "hybrid"
    if "attention" in roles and ("mla" in arch or any(section["label"] == "MLA" for section in sections)):
        return "mla"
    if "moe" in roles or "expert" in roles or "moe" in arch or "experts" in arch:
        return "moe"
    return "dense"


def label_for(section: dict[str, Any] | None, fallback: str) -> str:
    return section["label"] if section is not None else fallback


def target_id(target: dict[str, Any] | None) -> str | None:
    return target["id"] if target is not None else None


def target_section_id(target: dict[str, Any] | None) -> str | None:
    if target is None:
        return None
    return target.get("section_id", target["id"])


def anchor_by_role(
    anchors: list[dict[str, Any]],
    role: str,
    section: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return next(
        (
            anchor
            for anchor in anchors
            if anchor["role"] == role and (section is None or anchor["section_id"] == section["id"])
        ),
        None,
    )


def node(
    node_id: str,
    label: str,
    target: dict[str, Any] | None,
    role: str,
    x: int,
    y: int,
    w: int = 170,
    h: int = 42,
    *,
    shape: str = "box",
    rx: int = 10,
    subtitle: str | None = None,
    tone: str | None = None,
) -> dict[str, Any]:
    item = {
        "id": node_id,
        "label": label,
        "target_id": target_id(target),
        "section_id": target_section_id(target),
        "role": role,
        "shape": shape,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "rx": rx,
    }
    if subtitle is not None:
        item["subtitle"] = subtitle
    if tone is not None:
        item["tone"] = tone
    return item


def group(
    group_id: str,
    label: str,
    target: dict[str, Any] | None,
    role: str,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    rx: int = 24,
    badge: str | None = None,
    tone: str | None = None,
    outline: str | None = None,
    show_label: bool = True,
) -> dict[str, Any]:
    item = {
        "id": group_id,
        "label": label,
        "target_id": target_id(target),
        "section_id": target_section_id(target),
        "role": role,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "rx": rx,
        "badge": badge,
    }
    if tone is not None:
        item["tone"] = tone
    if outline is not None:
        item["outline"] = outline
    if not show_label:
        item["showLabel"] = False
    return item


def edge(
    edge_id: str,
    path: str,
    *,
    role: str = "flow",
    dashed: bool = False,
    arrow: bool = True,
    target: dict[str, Any] | None = None,
    tone: str | None = None,
) -> dict[str, Any]:
    item = {
        "id": edge_id,
        "path": path,
        "role": role,
        "dashed": dashed,
        "arrow": arrow,
        "target_id": target_id(target),
        "section_id": target_section_id(target),
    }
    if tone is not None:
        item["tone"] = tone
    return item


def rich_line(*parts: tuple[str, str | None] | str) -> list[dict[str, str]]:
    line: list[dict[str, str]] = []
    for part in parts:
        if isinstance(part, tuple):
            text, tone = part
        else:
            text, tone = part, None
        run = {"text": text}
        if tone is not None:
            run["tone"] = tone
        line.append(run)
    return line


def plain_lines(lines: list[list[dict[str, str]]]) -> str:
    return " ".join("".join(run["text"] for run in line) for line in lines)


def annotation(
    annotation_id: str,
    label: str,
    value: str,
    x: int,
    y: int,
    to_x: int,
    to_y: int,
    *,
    role: str = "metric",
    target: dict[str, Any] | None = None,
    side: str | None = None,
    lines: list[list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    item = {
        "id": annotation_id,
        "label": label,
        "value": value,
        "target_id": target_id(target),
        "section_id": target_section_id(target),
        "role": role,
        "x": x,
        "y": y,
        "to": {"x": to_x, "y": to_y},
    }
    if side is not None:
        item["side"] = side
    if lines is not None:
        item["lines"] = lines
        item["label"] = plain_lines(lines)
        item["value"] = ""
    return item


def rich_annotation(
    annotation_id: str,
    lines: list[list[dict[str, str]]],
    x: int,
    y: int,
    to_x: int,
    to_y: int,
    *,
    role: str = "metric",
    target: dict[str, Any] | None = None,
    side: str | None = None,
) -> dict[str, Any]:
    return annotation(
        annotation_id,
        plain_lines(lines),
        "",
        x,
        y,
        to_x,
        to_y,
        role=role,
        target=target,
        side=side,
        lines=lines,
    )


def decoration(
    decoration_id: str,
    role: str,
    *,
    text: str | None = None,
    path: str | None = None,
    x: int | None = None,
    y: int | None = None,
    lines: list[list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"id": decoration_id, "role": role}
    if text is not None:
        item["text"] = text
    if path is not None:
        item["path"] = path
    if x is not None:
        item["x"] = x
    if y is not None:
        item["y"] = y
    if lines is not None:
        item["lines"] = lines
    return item


def fmt_int(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def ffn_hidden_dimension(config: dict[str, Any]) -> Any:
    for key in ("dense_intermediate_size", "intermediate_size", "moe_intermediate_size"):
        if key in config:
            return config[key]
    if config.get("bias") is True and "n_embd" in config:
        return config["n_embd"] * 4
    return None


def config_value(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return None


def metric_annotations(
    config: dict[str, Any],
    model: dict[str, Any] | None,
    block_target: dict[str, Any] | None,
    attention: dict[str, Any] | None,
    embedding_target: dict[str, Any] | None = None,
    ffn_target: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    vocab = config_value(config, "vocab_size")
    if vocab is not None:
        items.append(
            rich_annotation(
                "metric-vocab",
                [rich_line("Vocabulary size of ", (fmt_int(vocab), "accent"))],
                615,
                128,
                405,
                171,
                target=model,
            )
        )
    context = config_value(config, "context_length")
    if context is not None:
        output_context = config_value(config, "output_context_length")
        value = (
            f"{fmt_int(context)} input / {fmt_int(output_context)} output"
            if output_context is not None
            else fmt_int(context)
        )
        items.append(
            rich_annotation(
                "metric-context",
                [
                    rich_line("Supported context"),
                    rich_line("length of ", (value, "accent")),
                    rich_line("tokens"),
                ],
                28,
                606,
                260,
                642,
                target=model,
                side="right",
            )
        )
    n_head = config_value(config, "n_head", "linear_n_head")
    n_kv_head = config_value(config, "n_kv_head")
    if n_head is not None:
        lines = (
            [rich_line((fmt_int(n_head), "accent"), " heads / ", (fmt_int(n_kv_head), "accent"), " KV heads")]
            if n_kv_head is not None
            else [rich_line((fmt_int(n_head), "accent"), " heads")]
        )
        items.append(rich_annotation("metric-heads", lines, 552, 548, 445, 474, target=attention))
    embedding_dimension = config_value(config, "n_embd", "hidden_size")
    if embedding_dimension is not None:
        items.append(
            rich_annotation(
                "metric-embedding",
                [rich_line("Embedding dimension of ", (fmt_int(embedding_dimension), "accent"))],
                552,
                632,
                450,
                642,
                target=embedding_target or model,
            )
        )
    hidden = ffn_hidden_dimension(config)
    if hidden is not None:
        items.append(
            rich_annotation(
                "metric-hidden",
                [rich_line("Hidden layer dimension of ", (fmt_int(hidden), "accent"))],
                552,
                690,
                430,
                344,
                target=ffn_target or model,
            )
        )
    return items


def add_ffn_detail(
    groups: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> None:
    mlp = first_section(sections, "mlp") or first_section(sections, "expert")
    gate = anchor_by_role(anchors, "mlp_gate", mlp)
    output = anchor_by_role(anchors, "mlp_output", mlp)
    groups.append(
        group(
            "ffn-detail",
            "SwiGLU / MLP detail",
            mlp,
            "detail",
            590,
            302,
            275,
            205,
            rx=18,
            outline="dotted",
            show_label=False,
        )
    )
    nodes.extend(
        [
            node("ffn-gate", "Gate projection", gate or mlp, "mlp", 620, 418, 110, 36),
            node("ffn-up", "Up projection", gate or mlp, "mlp", 738, 418, 110, 36),
            node("ffn-act", "SiLU / GELU", output or mlp, "mlp", 622, 362, 108, 36),
            node("ffn-down", "Down projection", output or mlp, "mlp", 680, 318, 125, 36),
        ]
    )
    edges.extend(
        [
            edge("ffn-callout", "M430 337 C505 337 545 350 590 365", role="callout", dashed=True, target=mlp),
            edge("ffn-gate-act", "M675 418 L675 398", role="detail", target=gate or mlp),
            edge("ffn-up-act", "M793 418 C793 388 760 380 730 380", role="detail", arrow=False, target=gate or mlp),
            edge("ffn-act-down", "M730 362 L744 354", role="detail", target=output or mlp),
        ]
    )


def add_moe_detail(
    groups: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    moe = first_section(sections, "moe")
    expert = first_section(sections, "expert") or moe
    router = anchor_by_role(anchors, "router", moe)
    topk = anchor_by_role(anchors, "topk", moe)
    experts = anchor_by_role(anchors, "experts", moe)
    shared = anchor_by_role(anchors, "shared_expert", moe)
    groups.append(
        group(
            "moe-detail",
            "MoE detail",
            moe,
            "moe",
            585,
            270,
            290,
            255,
            rx=18,
            tone="accent",
            outline="dotted",
            show_label=False,
        )
    )
    nodes.extend(
        [
            node("moe-router", "Router / gate", router or moe, "router", 620, 458, 120, 36),
            node("moe-topk", "Top-k select", topk or moe, "router", 620, 406, 120, 36),
            node("moe-experts", label_for(expert, "Experts"), experts or expert, "expert", 620, 342, 120, 42),
            node("moe-shared", "Shared expert", shared or moe, "expert", 742, 342, 110, 42),
        ]
    )
    edges.extend(
        [
            edge(
                "moe-callout",
                "M430 337 C520 337 535 380 585 395",
                role="callout",
                dashed=True,
                target=moe,
                tone="accent",
            ),
            edge("moe-router-topk", "M680 458 L680 442", role="detail", target=topk or moe),
            edge("moe-topk-experts", "M680 406 L680 384", role="detail", target=experts or expert),
            edge("moe-shared-join", "M742 363 L740 363", role="detail", arrow=False, target=shared or moe),
        ]
    )
    n_experts = config_value(config, "n_experts")
    top_k = config_value(config, "n_experts_per_tok")
    if n_experts is not None or top_k is not None:
        value = " / ".join(
            part
            for part in (
                f"{fmt_int(n_experts)} experts" if n_experts is not None else "",
                f"top-{fmt_int(top_k)}" if top_k is not None else "",
            )
            if part
        )
        nodes.append(node("moe-badge", value, moe, "badge", 690, 292, 145, 30, rx=15))


def add_mla_detail(
    groups: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    attention = first_section(sections, "attention")
    q_low = anchor_by_role(anchors, "latent_query", attention)
    kv_latent = anchor_by_role(anchors, "latent_kv", attention)
    rope = anchor_by_role(anchors, "rope", attention)
    output = anchor_by_role(anchors, "output_projection", attention)
    groups.append(
        group(
            "mla-detail",
            "MLA detail",
            attention,
            "attention",
            585,
            270,
            290,
            260,
            rx=18,
            outline="dotted",
            show_label=False,
        )
    )
    nodes.extend(
        [
            node("mla-query", "Query low-rank", q_low or attention, "attention", 618, 446, 125, 38),
            node("mla-kv", "Latent KV cache", kv_latent or attention, "latent", 618, 388, 125, 38),
            node("mla-rope", "Decoupled RoPE", rope or attention, "position", 730, 388, 125, 38),
            node("mla-out", "Output projection", output or attention, "attention", 675, 320, 130, 38),
        ]
    )
    edges.extend(
        [
            edge("mla-callout", "M430 477 C520 470 540 405 585 405", role="callout", dashed=True, target=attention),
            edge("mla-query-out", "M680 446 C585 430 580 365 675 358", role="detail", target=q_low or attention),
            edge("mla-kv-out", "M680 388 L700 358", role="detail", target=kv_latent or attention),
            edge("mla-rope-out", "M792 388 L755 358", role="detail", target=rope or attention),
        ]
    )
    kv_rank = config_value(config, "kv_lora_rank")
    q_rank = config_value(config, "q_lora_rank")
    if kv_rank is not None or q_rank is not None:
        label = " / ".join(
            part
            for part in (
                f"q rank {fmt_int(q_rank)}" if q_rank is not None else "",
                f"kv rank {fmt_int(kv_rank)}" if kv_rank is not None else "",
            )
            if part
        )
        nodes.append(node("mla-badge", label, attention, "badge", 650, 292, 170, 30, rx=15))


def add_mixer_detail(
    groups: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    mixer = first_section(sections, "mixer")
    state = anchor_by_role(anchors, "mixer_state", mixer)
    out = anchor_by_role(anchors, "output_projection", mixer)
    groups.append(
        group(
            "mixer-detail",
            "Mixer detail",
            mixer,
            "mixer",
            585,
            302,
            290,
            210,
            rx=18,
            outline="dotted",
            show_label=False,
        )
    )
    nodes.extend(
        [
            node("mixer-proj", "Q/K/V gates", anchor_by_role(anchors, "qkv", mixer) or mixer, "mixer", 622, 434, 120, 38),
            node("mixer-state", "State update", state or mixer, "mixer", 622, 376, 120, 38),
            node("mixer-out", "Read + output", out or mixer, "mixer", 700, 324, 130, 38),
        ]
    )
    edges.extend(
        [
            edge("mixer-callout", "M430 477 C520 477 540 405 585 405", role="callout", dashed=True, target=mixer),
            edge("mixer-proj-state", "M682 434 L682 414", role="detail", target=state or mixer),
            edge("mixer-state-out", "M742 376 L765 362", role="detail", target=out or mixer),
        ]
    )
    attn_every = config_value(config, "attn_every")
    if attn_every is not None:
        nodes.append(node("mixer-badge", f"attention every {fmt_int(attn_every)}", mixer, "badge", 650, 292, 170, 30, rx=15))


def add_sparse_attention_note(
    slug: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    attention: dict[str, Any] | None,
    config: dict[str, Any],
) -> None:
    labels = {
        "minimax-m3": "MiniMax sparse attn (full fallback)",
        "glm-5.2": "DSA / IndexShare (full fallback)",
        "glm-5": "Sparse attn omitted",
        "glm-5.1": "Sparse attn omitted",
        "deepseek-v3.2": "Sparse attn omitted",
        "deepseek-v4-flash": "Compressed sparse attn omitted",
        "deepseek-v4-pro": "Compressed sparse attn omitted",
    }
    if slug not in labels and "sparse_attention_topk_blocks" not in config:
        return
    target = anchor_by_role(anchors, "sparse_attention", attention) or attention
    label = labels.get(slug, "Sparse attention metadata")
    nodes.append(node("sparse-attn-note", label, target, "attention", 590, 220, 275, 36, rx=8))
    edges.append(edge("sparse-attn-callout", "M430 492 C520 470 540 238 590 238", role="callout", dashed=True, target=target))


def layout_column(
    items: list[dict[str, Any]],
    *,
    x: int,
    y_bottom: int,
    width: int,
    gap: int = 18,
    default_height: int = 42,
) -> dict[str, dict[str, Any]]:
    y = y_bottom
    laid_out: dict[str, dict[str, Any]] = {}
    for item in items:
        height = int(item.get("h", default_height))
        y -= height
        laid_out[item["id"]] = {**item, "x": x, "y": y, "w": int(item.get("w", width)), "h": height}
        y -= gap
    return laid_out


def center_x(box: dict[str, Any]) -> int:
    return int(box["x"] + box["w"] / 2)


def center_y(box: dict[str, Any]) -> int:
    return int(box["y"] + box["h"] / 2)


def top_y(box: dict[str, Any]) -> int:
    return int(box["y"])


def bottom_y(box: dict[str, Any]) -> int:
    return int(box["y"] + box["h"])


def flow_edge(edge_id: str, lower: dict[str, Any], upper: dict[str, Any], target: dict[str, Any] | None) -> dict[str, Any]:
    x1 = center_x(lower)
    x2 = center_x(upper)
    return edge(edge_id, f"M{x1} {top_y(lower)} L{x2} {bottom_y(upper)}", target=target)


def make_gpt2_diagram(
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    model = first_section(sections, "model")
    block = first_section(sections, "block")
    attention = first_section(sections, "attention")
    mlp = first_section(sections, "mlp")
    token_embedding = anchor_by_role(anchors, "embedding") or model
    position_embedding = anchor_by_role(anchors, "position_embedding") or token_embedding
    embedding_dropout = anchor_by_role(anchors, "embedding_dropout") or model
    block_stack = anchor_by_role(anchors, "block_stack") or block
    output_head = anchor_by_role(anchors, "output_head") or model
    final_norm = anchor_by_role(anchors, "final_norm") or output_head
    lm_head = anchor_by_role(anchors, "lm_head") or output_head
    attn_residual = anchor_by_role(anchors, "residual_attn", block) or block
    ffn_residual = anchor_by_role(anchors, "residual_mlp", block) or block
    attn_norm = anchor_by_role(anchors, "norm_1", block) or attn_residual
    ffn_norm = anchor_by_role(anchors, "norm_2", block) or ffn_residual
    attn_dropout = anchor_by_role(anchors, "attention_dropout", attention) or attention
    mlp_dropout = anchor_by_role(anchors, "mlp_dropout", mlp) or mlp
    mlp_input = anchor_by_role(anchors, "mlp_gate", mlp) or mlp
    mlp_activation = anchor_by_role(anchors, "activation", mlp) or mlp
    mlp_output = anchor_by_role(anchors, "mlp_output", mlp) or mlp

    groups = [
        group("model-shell", "Model shell", model, "model", 140, 70, 390, 690, rx=32),
        group(
            "block-shell",
            "Repeated block",
            block_stack,
            "block",
            210,
            225,
            250,
            400,
            rx=28,
            show_label=False,
        ),
        group(
            "ffn-detail",
            "MLP detail",
            mlp,
            "detail",
            590,
            288,
            275,
            205,
            rx=18,
            outline="dotted",
            show_label=False,
        ),
    ]
    nodes = [
        node(
            "input",
            "Tokenized text",
            model,
            "input",
            245,
            792,
            180,
            38,
            subtitle="Every effort moves you",
        ),
        node("embedding", "Token embedding layer", token_embedding, "embedding", 210, 722, 250, 42),
        node("position", "Positional embedding layer", position_embedding, "position", 195, 666, 280, 42),
        node("embed-dropout", "Dropout", embedding_dropout, "dropout", 250, 620, 160, 34),
        node("block-norm-1", "LayerNorm 1", attn_norm, "norm", 250, 570, 160, 38),
        node("compute", "Masked multi-head attention", attention, "attention", 230, 496, 205, 52, tone="dark"),
        node("attn-dropout", "Dropout", attn_dropout, "dropout", 250, 448, 160, 34),
        node("plus-1", "+", attn_residual, "plus", 315, 408, 32, 32, shape="circle"),
        node("block-norm-2", "LayerNorm 2", ffn_norm, "norm", 250, 360, 160, 38),
        node("feed-forward", "Feed forward", mlp, "mlp", 230, 304, 205, 48),
        node("ffn-dropout", "Dropout", mlp_dropout, "dropout", 250, 258, 160, 34),
        node("plus-2", "+", ffn_residual, "plus", 315, 220, 32, 32, shape="circle"),
        node("final-norm", "Final LayerNorm", final_norm, "norm", 245, 162, 175, 38),
        node("lm-head", "Linear output layer", lm_head, "output", 226, 102, 214, 38),
        node("ffn-linear-1", "Linear layer", mlp_input, "mlp", 650, 420, 125, 36),
        node("ffn-act", "GELU activation", mlp_activation, "mlp", 650, 360, 125, 36),
        node("ffn-linear-2", "Linear layer", mlp_output, "mlp", 650, 312, 125, 36),
    ]
    edges = [
        edge("flow-input-embedding", "M335 792 L335 764", target=token_embedding),
        edge("flow-embedding-position", "M335 722 L335 708", target=position_embedding),
        edge("flow-position-dropout", "M335 666 L335 654", target=embedding_dropout),
        edge("flow-dropout-norm", "M335 620 L335 608", target=block_stack),
        edge("flow-norm-compute", "M335 570 L335 548", target=attention),
        edge("flow-compute-dropout", "M335 496 L335 482", target=attn_dropout),
        edge("flow-dropout-plus", "M335 448 L335 440", target=attn_residual),
        edge("flow-plus-norm", "M335 408 L335 398", target=ffn_residual),
        edge("flow-norm-ffn", "M335 360 L335 352", target=mlp),
        edge("flow-ffn-dropout", "M335 304 L335 292", target=mlp_dropout),
        edge("flow-dropout-plus-2", "M335 258 L335 252", target=ffn_residual),
        edge("flow-plus-final", "M335 220 L335 200", target=output_head),
        edge("flow-final-head", "M335 162 L335 140", target=output_head),
        edge("flow-head-out", "M335 102 L335 78", target=output_head),
        edge("residual-attn", "M255 589 C205 589 205 424 315 424", role="residual", arrow=False, target=attn_residual),
        edge("residual-ffn", "M255 379 C205 379 205 236 315 236", role="residual", arrow=False, target=ffn_residual),
        edge("ffn-callout", "M435 328 C510 328 545 350 590 382", role="callout", dashed=True, arrow=False, target=mlp),
        edge("ffn-linear-act", "M712 420 L712 396", role="detail", target=mlp_activation),
        edge("ffn-act-linear", "M712 360 L712 348", role="detail", target=mlp_output),
    ]

    context_length = fmt_int(config.get("context_length", ""))
    annotations = [
        rich_annotation(
            "metric-vocab",
            [rich_line("Vocabulary size of ", (fmt_int(config.get("vocab_size", "")), "accent"))],
            510,
            44,
            405,
            121,
            target=model,
            side="bottom",
        ),
        rich_annotation(
            "metric-heads",
            [rich_line((fmt_int(config.get("n_head", "")), "accent"), " heads")],
            552,
            520,
            435,
            522,
            target=attention,
        ),
        rich_annotation(
            "metric-context-position",
            [
                rich_line("Supported context"),
                rich_line("length of ", (context_length, "accent")),
                rich_line("tokens with"),
                rich_line("absolute position"),
                rich_line("embeddings"),
            ],
            575,
            610,
            475,
            687,
            target=position_embedding,
            side="left",
        ),
        rich_annotation(
            "metric-context-token-input",
            [
                rich_line("Supported"),
                rich_line("context length"),
                rich_line("of ", (str(config.get("context_length", "")), "accent"), " tokens"),
            ],
            8,
            815,
            210,
            743,
            target=token_embedding,
            side="right",
        ),
        rich_annotation(
            "metric-embedding",
            [
                rich_line("Embedding"),
                rich_line("dimension of ", (fmt_int(config.get("n_embd", "")), "accent")),
            ],
            650,
            790,
            460,
            743,
            target=token_embedding,
            side="left",
        ),
        rich_annotation(
            "metric-hidden",
            [
                rich_line("Hidden layer"),
                rich_line("dimension of ", (fmt_int(ffn_hidden_dimension(config)), "accent")),
            ],
            650,
            535,
            720,
            493,
            target=mlp,
            side="top",
        ),
    ]
    decorations = [
        decoration(
            "repeat-brace",
            "repeat-brace",
            path="M205 560 C188 560 188 582 199 587 C188 592 188 614 205 614",
        ),
        decoration(
            "repeat-count",
            "repeat-count",
            x=150,
            y=600,
            lines=[rich_line((f"{fmt_int(config.get('n_layer', ''))} \u00d7", "accent"))],
        ),
    ]
    return {
        "template": "dense",
        "profile": "gpt2",
        "viewBox": "0 0 900 890",
        "groups": groups,
        "nodes": nodes,
        "edges": edges,
        "annotations": annotations,
        "decorations": decorations,
    }


def make_decoder_diagram(
    slug: str,
    template: str,
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if slug == "gpt2-xl":
        return make_gpt2_diagram(sections, anchors, config)

    model = first_section(sections, "model")
    block = first_section(sections, "block")
    attention = first_section(sections, "attention")
    mixer = first_section(sections, "mixer")
    mlp = first_section(sections, "mlp")
    moe = first_section(sections, "moe")
    position = first_section(sections, "position")
    embedding = anchor_by_role(anchors, "embedding") or model
    block_stack = anchor_by_role(anchors, "block_stack") or block
    output_head = anchor_by_role(anchors, "output_head") or model
    final_norm = anchor_by_role(anchors, "final_norm") or output_head
    lm_head = anchor_by_role(anchors, "lm_head") or output_head
    attn_residual = anchor_by_role(anchors, "residual_attn", block) or block
    ffn_residual = anchor_by_role(anchors, "residual_mlp", block) or block
    attn_norm = anchor_by_role(anchors, "norm_1", block) or attn_residual
    ffn_norm = anchor_by_role(anchors, "norm_2", block) or ffn_residual
    qk_norm = anchor_by_role(anchors, "qk_norm", attention)
    rope = anchor_by_role(anchors, "rope", attention) or position
    compute = mixer if template == "hybrid" and mixer is not None else attention
    compute_role = "mixer" if template == "hybrid" and mixer is not None else "attention"
    compute_label = label_for(compute, "Sequence mixer" if compute_role == "mixer" else "Attention")
    ffn_target = moe or mlp
    ffn_role = "moe" if moe is not None else "mlp"
    ffn_label = label_for(ffn_target, "Feed forward")
    if template == "mla":
        compute_label = label_for(attention, "Multi-head Latent Attention")
    if template == "parallel":
        ffn_label = "Parallel MoE" if moe is not None else "Parallel FFN"

    stack = layout_column(
        [
            {"id": "input", "h": 38, "w": 180},
            {"id": "embedding"},
            {"id": "block-norm-1", "h": 38, "w": 160},
            {"id": "compute", "h": 52, "w": 205},
            {"id": "plus-1", "h": 32, "w": 32},
            {"id": "block-norm-2", "h": 38, "w": 160},
            {"id": "feed-forward", "h": 48, "w": 205},
            {"id": "plus-2", "h": 32, "w": 32},
            {"id": "final-norm", "h": 38, "w": 175},
            {"id": "lm-head", "h": 38, "w": 214},
        ],
        x=335,
        y_bottom=756,
        width=250,
        gap=18,
    )
    for item in stack.values():
        item["x"] = int(335 - item["w"] / 2)

    groups = [
        group("model-shell", "Model shell", model, "model", 140, 86, 390, 610, rx=32),
        group(
            "block-shell",
            "Repeated block",
            block_stack,
            "block",
            210,
            top_y(stack["block-norm-1"]) - 28,
            250,
            bottom_y(stack["plus-2"]) - top_y(stack["block-norm-1"]) + 56,
            rx=28,
            show_label=False,
        ),
    ]
    nodes = [
        node(
            "input",
            "Tokenized text",
            model,
            "input",
            stack["input"]["x"],
            stack["input"]["y"],
            stack["input"]["w"],
            stack["input"]["h"],
            subtitle="Every effort moves you",
        ),
        node("embedding", "Token embedding", embedding, "embedding", stack["embedding"]["x"], stack["embedding"]["y"], stack["embedding"]["w"], stack["embedding"]["h"]),
        node("block-norm-1", "Norm 1", attn_norm, "norm", stack["block-norm-1"]["x"], stack["block-norm-1"]["y"], stack["block-norm-1"]["w"], stack["block-norm-1"]["h"]),
        node(
            "compute",
            compute_label,
            compute,
            compute_role,
            stack["compute"]["x"],
            stack["compute"]["y"],
            stack["compute"]["w"],
            stack["compute"]["h"],
            tone="dark" if compute_role == "attention" else None,
        ),
        node("plus-1", "+", attn_residual, "plus", stack["plus-1"]["x"], stack["plus-1"]["y"], stack["plus-1"]["w"], stack["plus-1"]["h"], shape="circle"),
        node("block-norm-2", "Norm 2", ffn_norm, "norm", stack["block-norm-2"]["x"], stack["block-norm-2"]["y"], stack["block-norm-2"]["w"], stack["block-norm-2"]["h"]),
        node("feed-forward", ffn_label, ffn_target, ffn_role, stack["feed-forward"]["x"], stack["feed-forward"]["y"], stack["feed-forward"]["w"], stack["feed-forward"]["h"]),
        node("plus-2", "+", ffn_residual, "plus", stack["plus-2"]["x"], stack["plus-2"]["y"], stack["plus-2"]["w"], stack["plus-2"]["h"], shape="circle"),
        node("final-norm", "Final norm", final_norm, "norm", stack["final-norm"]["x"], stack["final-norm"]["y"], stack["final-norm"]["w"], stack["final-norm"]["h"]),
        node("lm-head", "Linear output layer", lm_head, "output", stack["lm-head"]["x"], stack["lm-head"]["y"], stack["lm-head"]["w"], stack["lm-head"]["h"]),
    ]
    if template == "parallel":
        nodes[2] = node("block-norm-1", "Shared norm", anchor_by_role(anchors, "parallel", block) or block, "norm", stack["block-norm-1"]["x"], stack["block-norm-1"]["y"], stack["block-norm-1"]["w"], stack["block-norm-1"]["h"])
        nodes[4] = node("plus-1", "+", anchor_by_role(anchors, "parallel", block) or block, "plus", stack["plus-1"]["x"], stack["plus-1"]["y"], stack["plus-1"]["w"], stack["plus-1"]["h"], shape="circle")
        nodes[5] = node("parallel-branch", "Parallel sum", anchor_by_role(anchors, "parallel", block) or block, "block", stack["block-norm-2"]["x"], stack["block-norm-2"]["y"], stack["block-norm-2"]["w"], stack["block-norm-2"]["h"])

    if position is not None:
        nodes.append(node("position", label_for(position, "Position signal"), rope or position, "position", 68, stack["compute"]["y"] + 12, 135, 38))
    if qk_norm is not None:
        nodes.append(node("qk-norm", "Q/K norm", qk_norm, "norm", 55, stack["plus-1"]["y"] - 5, 145, 38))

    edges = [
        flow_edge("flow-input-embedding", stack["input"], stack["embedding"], embedding),
        flow_edge("flow-embedding-norm", stack["embedding"], stack["block-norm-1"], block_stack),
        flow_edge("flow-norm-compute", stack["block-norm-1"], stack["compute"], compute),
        flow_edge("flow-compute-plus", stack["compute"], stack["plus-1"], attn_residual),
        flow_edge("flow-plus-norm", stack["plus-1"], stack["block-norm-2"], ffn_residual),
        flow_edge("flow-norm-ffn", stack["block-norm-2"], stack["feed-forward"], ffn_target),
        flow_edge("flow-ffn-plus", stack["feed-forward"], stack["plus-2"], ffn_residual),
        flow_edge("flow-plus-final", stack["plus-2"], stack["final-norm"], output_head),
        flow_edge("flow-final-head", stack["final-norm"], stack["lm-head"], output_head),
        edge("flow-head-out", f"M335 {top_y(stack['lm-head'])} L335 116", target=output_head),
        edge(
            "residual-attn",
            f"M255 {center_y(stack['block-norm-1'])} C205 {center_y(stack['block-norm-1'])} 205 {center_y(stack['plus-1'])} {stack['plus-1']['x']} {center_y(stack['plus-1'])}",
            role="residual",
            arrow=False,
            target=attn_residual,
        ),
        edge(
            "residual-ffn",
            f"M255 {center_y(stack['block-norm-2'])} C205 {center_y(stack['block-norm-2'])} 205 {center_y(stack['plus-2'])} {stack['plus-2']['x']} {center_y(stack['plus-2'])}",
            role="residual",
            arrow=False,
            target=ffn_residual,
        ),
    ]
    if position is not None:
        edges.append(edge("position-to-compute", f"M203 {stack['compute']['y'] + 31} L230 {stack['compute']['y'] + 26}", role="side", target=rope or position))
    if qk_norm is not None:
        edges.append(edge("qk-to-compute", f"M200 {stack['plus-1']['y'] + 14} L230 {stack['compute']['y'] + 34}", role="side", target=qk_norm))

    annotations = metric_annotations(config, model, block_stack, compute, embedding, ffn_target)
    decorations = []
    if "n_layer" in config:
        decorations.extend(
            [
                decoration(
                    "repeat-brace",
                    "repeat-brace",
                    path=(
                        f"M205 {top_y(stack['block-norm-1']) - 16} "
                        f"C188 {top_y(stack['block-norm-1']) - 16} 188 {top_y(stack['block-norm-1']) + 6} "
                        f"199 {top_y(stack['block-norm-1']) + 11} C188 {top_y(stack['block-norm-1']) + 16} "
                        f"188 {top_y(stack['block-norm-1']) + 38} 205 {top_y(stack['block-norm-1']) + 38}"
                    ),
                ),
                decoration(
                    "repeat-count",
                    "repeat-count",
                    x=150,
                    y=top_y(stack["block-norm-1"]) + 24,
                    lines=[rich_line((f"{fmt_int(config['n_layer'])} \u00d7", "accent"))],
                ),
            ]
        )
    if template == "moe" or (template == "parallel" and moe is not None):
        add_moe_detail(groups, nodes, edges, sections, anchors, config)
    elif template == "hybrid":
        add_mixer_detail(groups, nodes, edges, sections, anchors, config)
    elif template != "mla":
        add_ffn_detail(groups, nodes, edges, sections, anchors)

    if template == "mla":
        add_mla_detail(groups, nodes, edges, sections, anchors, config)

    add_sparse_attention_note(slug, nodes, edges, anchors, attention, config)

    return {
        "template": template,
        "viewBox": "0 0 900 820",
        "groups": groups,
        "nodes": nodes,
        "edges": edges,
        "annotations": annotations,
        "decorations": decorations,
    }


def make_diagram(
    slug: str,
    template: str,
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return make_decoder_diagram(slug, template, sections, anchors, config)


def diagram_palette(slug: str) -> dict[str, str]:
    return dict(DIAGRAM_PALETTES[slug])


def load_architecture_manifest() -> dict[str, Any]:
    if not ARCHITECTURE_MANIFEST_PATH.exists():
        return {}
    return json.loads(ARCHITECTURE_MANIFEST_PATH.read_text(encoding="utf-8")).get("assets", {})


def add_artwork_metadata(slug: str, diagram: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    artwork = manifest.get(slug)
    if not artwork:
        return diagram
    diagram["artwork"] = {
        key: artwork[key]
        for key in (
            "path",
            "source_url",
            "source_title",
            "source_alt",
            "article_url",
            "width",
            "height",
            "source_sha256",
        )
        if key in artwork
    }
    return diagram


def load_hotspot_source(slug: str) -> dict[str, Any] | None:
    path = HOTSPOTS_DIR / f"{slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_hotspot_target(
    slug: str,
    hotspot: dict[str, Any],
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    def section_by_label(label: str | None) -> dict[str, Any] | None:
        match = next((section for section in sections if section["label"] == label), None)
        if match is not None:
            return match
        if label == "Block":
            return next((section for section in sections if section["role"] == "block"), None)
        return None

    target = hotspot.get("target", {})
    target_type = target.get("type")
    if target_type == "section":
        label = target.get("label")
        match = section_by_label(label)
        if match is None:
            raise ValueError(f"{slug}:{hotspot.get('id')} references unknown section label {label!r}")
        return match
    if target_type == "anchor":
        section_label = target.get("section_label")
        role = target.get("role")
        section = section_by_label(section_label)
        if section is None:
            raise ValueError(
                f"{slug}:{hotspot.get('id')} references unknown anchor section {section_label!r}"
            )
        match = next(
            (
                anchor
                for anchor in anchors
                if anchor["section_id"] == section["id"] and anchor["role"] == role
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"{slug}:{hotspot.get('id')} references unknown anchor role {role!r} "
                f"in section {section_label!r}"
            )
        return match
    raise ValueError(f"{slug}:{hotspot.get('id')} has unsupported hotspot target type {target_type!r}")


def add_hotspot_metadata(
    slug: str,
    diagram: dict[str, Any],
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    source = load_hotspot_source(slug)
    if source is None:
        return diagram
    if source.get("slug") != slug:
        raise ValueError(f"{slug} hotspot source has mismatched slug {source.get('slug')!r}")
    if source.get("coordinate_space") != "artwork":
        raise ValueError(f"{slug} hotspot source must use artwork coordinate_space")

    artwork = diagram.get("artwork")
    if not artwork:
        return diagram
    if source.get("image_sha256") != artwork.get("source_sha256"):
        raise ValueError(f"{slug} hotspot source image_sha256 does not match artwork")
    if source.get("checked") is not True:
        return diagram

    seen_ids: set[str] = set()
    hotspots: list[dict[str, Any]] = []
    for item in source.get("hotspots", []):
        hotspot_id = item.get("id")
        if not hotspot_id:
            raise ValueError(f"{slug} hotspot source contains an item without id")
        if hotspot_id in seen_ids:
            raise ValueError(f"{slug} hotspot source contains duplicate id {hotspot_id!r}")
        seen_ids.add(hotspot_id)
        if item.get("source") == "generated-template":
            raise ValueError(f"{slug}:{hotspot_id} uses generated-template source in a checked file")
        if item.get("source") not in {"manual", "ocr", "detected"}:
            raise ValueError(f"{slug}:{hotspot_id} has unsupported source {item.get('source')!r}")

        x, y, width, height = (int(item[key]) for key in ("x", "y", "w", "h"))
        if not (0 <= x < artwork["width"] and 0 <= y < artwork["height"]):
            raise ValueError(f"{slug}:{hotspot_id} starts outside artwork bounds")
        if not (1 <= width <= artwork["width"] - x and 1 <= height <= artwork["height"] - y):
            raise ValueError(f"{slug}:{hotspot_id} extends outside artwork bounds")
        shape = item.get("shape", "rect")
        if shape not in {"rect", "roundrect"}:
            raise ValueError(f"{slug}:{hotspot_id} has unsupported shape {shape!r}")

        target = resolve_hotspot_target(slug, item, sections, anchors)
        target_section = target_section_id(target)
        hotspot = {
            "id": hotspot_id,
            "label": item.get("label") or target.get("label", hotspot_id),
            "role": item.get("role") or target.get("role", "helper"),
            "target_id": target["id"],
            "section_id": target_section,
            "source": item["source"],
            "shape": shape,
            "x": x,
            "y": y,
            "w": width,
            "h": height,
        }
        if shape == "roundrect":
            hotspot["rx"] = int(item.get("rx", 0))
            hotspot["ry"] = int(item.get("ry", hotspot["rx"]))
        hotspots.append(hotspot)

    if hotspots:
        diagram["hotspots"] = hotspots
        diagram["hotspotsCoordinateSpace"] = "artwork"
    return diagram


def role_counts(sections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for section in sections:
        role = section["role"]
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: CORE_ROLE_ORDER.get(item[0], 99)))


def gallery_card_url(base_url: str, gallery_card_id: str) -> str:
    return f"{base_url.rstrip('/')}/#card-{gallery_card_id}"


def gallery_url_for(entry: registry.Entry, constants: dict[str, Any]) -> str:
    article_url = GALLERY_ARTICLE_URLS.get(entry.slug)
    if article_url:
        return article_url
    base_url = constants.get("GALLERY_URL") or GALLERY_BASE_URL
    gallery_card_id = GALLERY_SOURCE_KEYS.get(entry.slug, entry.slug)
    return gallery_card_url(base_url, gallery_card_id)


CONCEPT_REQUIRED_FIELDS = ("title", "emoji", "summary")
CONCEPT_DOLLAR_MARKER = "KATEXDOLLARMARKER"
CONCEPT_FENCE_RE = re.compile(r"(```.*?```)", re.DOTALL)
CONCEPT_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
CONCEPT_INLINE_MATH_RE = re.compile(r"\$([^$\n]+?)\$")


def render_concept_body(body: str, path: Path) -> str:
    """Convert concept markdown to HTML with $...$/$$...$$ preserved for client-side KaTeX.

    Math is extracted before the markdown conversion (so underscores/asterisks inside TeX
    survive) and re-inserted as .katex-src placeholder elements that web/app.js renders
    with katex.render(). Fenced code blocks are shielded so a stray $ inside them stays text.
    """
    stash: list[tuple[str, bool]] = []

    def stash_math(text: str) -> str:
        text = text.replace(r"\$", CONCEPT_DOLLAR_MARKER)

        def displaced(match: re.Match[str]) -> str:
            stash.append((match.group(1).strip(), True))
            return f"KATEXMATH{len(stash) - 1}MARKER"

        def inlined(match: re.Match[str]) -> str:
            stash.append((match.group(1).strip(), False))
            return f"KATEXMATH{len(stash) - 1}MARKER"

        text = CONCEPT_DISPLAY_MATH_RE.sub(displaced, text)
        text = CONCEPT_INLINE_MATH_RE.sub(inlined, text)
        if "$" in text:
            raise ValueError(f"{path.name}: unbalanced $ delimiter (escape literal dollars as \\$)")
        return text

    parts = CONCEPT_FENCE_RE.split(body)
    for i in range(0, len(parts), 2):
        parts[i] = stash_math(parts[i])
    rendered = markdown.markdown("".join(parts), extensions=["tables", "fenced_code"])

    for i, (tex, display) in enumerate(stash):
        marker = f"KATEXMATH{i}MARKER"
        escaped = html.escape(tex, quote=True)
        if display:
            element = f'<div class="katex-src" data-display="1" data-tex="{escaped}"></div>'
            rendered = rendered.replace(f"<p>{marker}</p>", element).replace(marker, element)
        else:
            rendered = rendered.replace(marker, f'<span class="katex-src" data-tex="{escaped}"></span>')
    rendered = rendered.replace(CONCEPT_DOLLAR_MARKER, "$")
    if "KATEXMATH" in rendered or "KATEXDOLLAR" in rendered:
        raise ValueError(f"{path.name}: math placeholder leaked into rendered HTML")
    return rendered


def parse_concept_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError(f"{path.name}: missing --- frontmatter fences")
    end = text.index("\n---\n", 4)
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"{path.name}: bad frontmatter line {line!r}")
        meta[key.strip()] = value.strip()
    missing = [field for field in CONCEPT_REQUIRED_FIELDS if not meta.get(field)]
    if missing:
        raise ValueError(f"{path.name}: missing frontmatter fields {missing}")
    related = [item.strip() for item in meta.get("related", "").split(",") if item.strip()]
    return {
        "id": path.stem,
        "title": meta["title"],
        "emoji": meta["emoji"],
        "summary": meta["summary"],
        "related": related,
        "body_html": render_concept_body(text[end + 5 :], path),
    }


def load_concepts() -> list[dict[str, Any]]:
    paths = sorted(CONCEPTS_DIR.glob("*.md"))
    if not paths:
        raise ValueError(f"no concept files found in {CONCEPTS_DIR}")
    concepts = [parse_concept_file(path) for path in paths]
    ids = {concept["id"] for concept in concepts}
    for concept in concepts:
        unknown = [rel for rel in concept["related"] if rel not in ids]
        if unknown:
            raise ValueError(f"{concept['id']}: unknown related concepts {unknown}")
    return concepts


def resolve_section_concept(section: dict[str, Any], section_anchors: list[dict[str, Any]]) -> str:
    label = section["label"]
    if label in SECTION_LABEL_CONCEPTS:
        return SECTION_LABEL_CONCEPTS[label]
    role = section["role"]
    if role == "attention":
        anchor_roles = {anchor["role"] for anchor in section_anchors}
        if "latent_kv" in anchor_roles:
            return "mla"
        if "kv_share" in anchor_roles:
            return "gqa"
        return "attention"
    if role in SECTION_ROLE_CONCEPTS:
        return SECTION_ROLE_CONCEPTS[role]
    raise ValueError(f"no concept mapping for section {section['id']} (role={role}, label={label})")


def resolve_anchor_concept(
    anchor: dict[str, Any],
    sections: list[dict[str, Any]],
    section_concepts: dict[str, str],
) -> str:
    role = anchor["role"]
    if role in ("norm_1", "norm_2", "final_norm"):
        norm = next((s for s in sections if s["role"] == "norm"), None)
        if norm is None:
            raise ValueError(f"anchor {anchor['id']} needs a norm section to inherit from")
        return section_concepts[norm["id"]]
    if role == "mixer_state":
        parent = next(s for s in sections if s["id"] == anchor["section_id"])
        if parent["role"] in ("mixer", "attention"):
            return section_concepts[parent["id"]]
        mixer = next((s for s in sections if s["role"] == "mixer"), None)
        if mixer is None:
            raise ValueError(f"anchor {anchor['id']} needs a mixer section to inherit from")
        return section_concepts[mixer["id"]]
    if role in ("mlp_gate", "mlp_output"):
        parent = next(s for s in sections if s["id"] == anchor["section_id"])
        if parent["role"] in ("mlp", "helper"):
            return section_concepts[parent["id"]]
        return "glu-feedforward"
    if role in ANCHOR_ROLE_CONCEPTS:
        return ANCHOR_ROLE_CONCEPTS[role]
    raise ValueError(f"no concept mapping for anchor {anchor['id']} (role={role})")


def assign_concepts(sections: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> None:
    section_concepts: dict[str, str] = {}
    for section in sections:
        section_anchors = [a for a in anchors if a["section_id"] == section["id"]]
        section["concept_id"] = resolve_section_concept(section, section_anchors)
        section_concepts[section["id"]] = section["concept_id"]
    for anchor in anchors:
        anchor["concept_id"] = resolve_anchor_concept(anchor, sections, section_concepts)


def build_model_payload(entry: registry.Entry, architecture_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    path = MODELS_DIR / f"{entry.module}.py"
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    constants = literal_constants(tree)
    sections = extract_sections(tree, source_lines)
    anchors = extract_anchors(tree, source_lines, sections)
    assign_concepts(sections, anchors)
    config = extract_config_defaults(tree)
    template = infer_template(sections, entry.archetype)
    diagram = make_diagram(entry.slug, template, sections, anchors, config)
    diagram["palette"] = diagram_palette(entry.slug)
    diagram = add_artwork_metadata(entry.slug, diagram, architecture_manifest or {})
    diagram = add_hotspot_metadata(entry.slug, diagram, sections, anchors)
    docstring = ast.get_docstring(tree, clean=True) or ""
    gallery_card_id = GALLERY_SOURCE_KEYS.get(entry.slug, entry.slug)
    gallery_url = gallery_url_for(entry, constants)

    return {
        "slug": entry.slug,
        "module": entry.module,
        "name": entry.name,
        "release": entry.release,
        "archetype": entry.archetype,
        "status": entry.status,
        "tier": entry.tier,
        "template": template,
        "source_path": str(path.relative_to(ROOT)),
        "source_lines": source_lines,
        "source_tokens": highlighted_source_lines(source),
        "line_count": len(source_lines),
        "summary": docstring.splitlines()[0] if docstring else entry.archetype,
        "config": config,
        "gallery_card_id": gallery_card_id,
        "notes": extract_marked_notes(tree),
        "links": {
            "gallery": gallery_url,
            "tech_report": constants.get("TECH_REPORT_URL", ""),
        },
        "sections": sections,
        "anchors": anchors,
        "section_role_counts": role_counts(sections),
        "diagram": diagram,
    }


def parameter_scale(name: str) -> str:
    match = re.search(r"\(([^)]*(?:B|M|T|A\d+)[^)]*)\)", name)
    return match.group(1) if match else ""


def release_year(release: str) -> str:
    return release[:4] if re.match(r"\d{4}", release or "") else ""


def index_entry(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "slug": payload["slug"],
        "module": payload["module"],
        "name": payload["name"],
        "release": payload["release"],
        "archetype": payload["archetype"],
        "status": payload["status"],
        "tier": payload["tier"],
        "template": payload["template"],
        "summary": payload["summary"],
        "gallery_card_id": payload["gallery_card_id"],
        "parameter_scale": parameter_scale(payload["name"]),
        "release_year": release_year(payload["release"]),
        "line_count": payload["line_count"],
        "section_role_counts": payload["section_role_counts"],
    }


def data_version_for(models: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for payload in models:
        digest.update(json_text(payload).encode("utf-8"))
    return digest.hexdigest()[:12]


def build_payloads() -> dict[str, Any]:
    architecture_manifest = load_architecture_manifest()
    concepts = load_concepts()
    concept_ids = {concept["id"] for concept in concepts}
    models = [build_model_payload(entry, architecture_manifest) for entry in registry.REGISTRY]
    for payload in models:
        for target in [*payload["sections"], *payload["anchors"]]:
            if target["concept_id"] not in concept_ids:
                raise ValueError(
                    f"{payload['slug']}: {target['id']} maps to missing concept "
                    f"{target['concept_id']!r} (add llm_gallery/concepts/{target['concept_id']}.md)"
                )
    concepts_payload = {
        "generated_by": "scripts/build_visualizer_data.py",
        "concept_count": len(concepts),
        "concepts": concepts,
    }
    data_version = data_version_for([*models, concepts_payload])
    for payload in models:
        payload["data_version"] = data_version
    concepts_payload["data_version"] = data_version
    index_models = [index_entry(payload) for payload in models]
    return {
        "index.json": {
            "generated_by": "scripts/build_visualizer_data.py",
            "data_version": data_version,
            "model_count": len(models),
            "templates": sorted({payload["template"] for payload in models}),
            "role_labels": ROLE_LABELS,
            "models": index_models,
        },
        "concepts.json": concepts_payload,
        **{f"{payload['slug']}.json": payload for payload in models},
    }


def json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def write_payloads(payloads: dict[str, Any], check: bool = False) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    expected = {DATA_DIR / filename: json_text(payload) for filename, payload in payloads.items()}

    if check:
        stale: list[str] = []
        for path, text in expected.items():
            if not path.exists():
                stale.append(f"missing {path.relative_to(ROOT)}")
            elif path.read_text(encoding="utf-8") != text:
                stale.append(f"stale {path.relative_to(ROOT)}")
        extra = sorted(path for path in DATA_DIR.glob("*.json") if path not in expected)
        stale.extend(f"extra {path.relative_to(ROOT)}" for path in extra)
        if stale:
            print("Visualizer data is not up to date:")
            for item in stale:
                print(f"  - {item}")
            print("Run: uv run python scripts/build_visualizer_data.py")
            return 1
        print("Visualizer data is up to date.")
        return 0

    for path, text in expected.items():
        path.write_text(text, encoding="utf-8")

    for path in DATA_DIR.glob("*.json"):
        if path not in expected:
            path.unlink()

    print(f"Wrote {len(expected)} JSON files to {DATA_DIR.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed JSON is stale")
    args = parser.parse_args(argv)
    return write_payloads(build_payloads(), check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
