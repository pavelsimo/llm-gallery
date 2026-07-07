from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_gallery.models import registry
from scripts.audit_fidelity import FIELD_DIFF_ALLOWLIST, field_diffs_for, real_preset

FIXTURES = Path(__file__).parent / "fixtures"
HF_CONFIGS = FIXTURES / "hf_configs"
EXPECTED_PARAM_COUNTS = json.loads((FIXTURES / "real_param_counts.json").read_text())

DECOUPLED_HEAD_DIM = {
    "command-a-plus",
    "gemma3-270m",
    "gemma3-27b",
    "gemma4-12b",
    "gemma4-26b-a4b",
    "gemma4-31b",
    "gemma4-e2b",
    "gemma4-e4b",
    "glm-4.5",
    "glm-4.5-air",
    "glm-4.7",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "hunyuan-3-preview",
    "intellect-3",
    "laguna-xs.2",
    "lfm2.5-1.2b",
    "lfm2.5-350m",
    "lfm2.5-8b-a1b",
    "mellum2-thinking",
    "mimo-v2-flash",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "minimax-m2",
    "minimax-m2.5",
    "minimax-m2.7",
    "minimax-m3",
    "mistral-small-3.1",
    "nemotron3-nano-30b",
    "nemotron3-nano-4b",
    "north-mini-code",
    "qwen3-0.6b",
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-32b",
    "qwen3-4b",
    "qwen3-coder-flash",
    "qwen3-next-80b-a3b",
    "qwen3.5",
    "qwen3.6-27b",
    "qwen3.6-35b-a3b",
    "step-3.5-flash",
    "zaya1-8b",
}


def done_slugs() -> list[str]:
    return registry.done_slugs()


@pytest.mark.slow
@pytest.mark.parametrize("slug", done_slugs())
def test_real_param_count_matches_fixture(slug: str):
    mod = registry.load(slug)
    _, cfg = real_preset(mod)
    with torch.device("meta"):
        model = mod.Model(cfg)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAM_COUNTS[slug]


@pytest.mark.parametrize("slug", sorted(p.stem for p in HF_CONFIGS.glob("*.json")))
def test_hf_config_field_mapping_matches_fixture(slug: str):
    assert not field_diffs_for(slug, HF_CONFIGS, {})


@pytest.mark.parametrize("slug", sorted(FIELD_DIFF_ALLOWLIST))
def test_hf_config_field_mapping_allowlist_is_still_needed(slug: str):
    fields = {diff.field for diff in field_diffs_for(slug, HF_CONFIGS, {}, include_allowed=True)}
    assert FIELD_DIFF_ALLOWLIST[slug] <= fields


@pytest.mark.parametrize("slug", done_slugs())
def test_real_preset_name_matches_slug(slug: str):
    mod = registry.load(slug)
    preset_name, _ = real_preset(mod)
    assert preset_name == slug


@pytest.mark.parametrize("slug", done_slugs())
def test_kv_heads_evenly_partition_query_heads(slug: str):
    mod = registry.load(slug)
    _, cfg = real_preset(mod)
    if not (hasattr(cfg, "n_head") and hasattr(cfg, "n_kv_head")):
        pytest.skip("model does not expose attention heads")
    assert cfg.n_head % cfg.n_kv_head == 0


@pytest.mark.parametrize("slug", done_slugs())
def test_head_dim_matches_width_unless_explicitly_decoupled(slug: str):
    mod = registry.load(slug)
    _, cfg = real_preset(mod)
    if not all(hasattr(cfg, name) for name in ("head_dim", "n_head", "n_embd")):
        pytest.skip("model does not expose head_dim/n_head/n_embd")

    is_decoupled = cfg.head_dim * cfg.n_head != cfg.n_embd
    assert is_decoupled == (slug in DECOUPLED_HEAD_DIM)


def test_wave2_archetype_fidelity_hooks():
    nemotron = registry.load("nemotron3-nano-30b")
    qwen_next = registry.load("qwen3-next-80b-a3b")
    llama4 = registry.load("llama4-maverick")
    tiny_aya = registry.load("tiny-aya")
    olmo3 = registry.load("olmo3-7b")
    step = registry.load("step-3.5-flash")
    laguna = registry.load("laguna-xs.2")

    assert nemotron.PRESETS["nemotron3-nano-30b"].layer_pattern == (
        "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    )
    tiny_nemotron = nemotron.Model(nemotron.PRESETS["tiny"])
    assert [block.layer_type for block in tiny_nemotron.blocks] == list("ME*MEM")

    qwen_cfg = qwen_next.PRESETS["qwen3-next-80b-a3b"]
    assert qwen_cfg.head_dim == 256
    assert qwen_cfg.rope_theta == 10_000_000.0
    assert qwen_cfg.partial_rotary_factor == 0.25
    assert qwen_cfg.linear_conv_kernel_dim == 4
    tiny_delta = qwen_next.GatedDeltaNet(qwen_next.PRESETS["tiny"])
    assert hasattr(tiny_delta, "q_conv")
    assert hasattr(tiny_delta, "k_conv")
    assert hasattr(tiny_delta, "v_conv")

    llama_cfg = llama4.PRESETS["llama4-maverick"]
    assert llama_cfg.moe_every == 2
    assert llama_cfg.nope_every == 4
    tiny_llama = llama4.Model(llama4.PRESETS["tiny"])
    assert [block.is_moe for block in tiny_llama.blocks] == [True, False, True, False]
    assert [block.attn.use_rope for block in tiny_llama.blocks] == [True, True, True, False]

    olmo_cfg = olmo3.PRESETS["olmo3-7b"]
    assert olmo_cfg.sliding_window == 4096
    assert olmo_cfg.global_every == 4
    tiny_olmo = olmo3.Model(olmo3.PRESETS["tiny"])
    assert [block.is_global for block in tiny_olmo.blocks] == [False, False, False, True]

    for mod in (step, laguna):
        cfg = mod.PRESETS[mod.DEFAULT_PRESET]
        assert cfg.sliding_window == 512
        assert cfg.global_every == 4
        assert cfg.global_first is True
        assert cfg.attention_gate is True
        tiny = mod.Model(mod.PRESETS["tiny"])
        assert [block.is_global for block in tiny.blocks[:4]] == [True, False, False, False]
        assert tiny.blocks[0].attn.g_proj is not None

    aya_cfg = tiny_aya.PRESETS["tiny-aya"]
    assert (aya_cfg.n_layer, aya_cfg.n_embd, aya_cfg.n_head, aya_cfg.n_kv_head) == (36, 2048, 16, 4)
    assert aya_cfg.intermediate_size == 11008
