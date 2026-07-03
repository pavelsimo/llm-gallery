"""Audit each model's ``real`` preset against published gallery/HF metadata.

Usage:
    uv run python scripts/audit_fidelity.py \
        --cards /tmp/gallery_cards.json --mapping /tmp/slug_to_card.json \
        --configs /tmp/hf_configs

Inputs are produced by scraping the LLM Architecture Gallery page (cards + slug
mapping) and downloading the per-model HF config.json files it links. The script
is also importable by tests: field mapping, context conventions, and meta-device
parameter counting all live here so the audit and regression harness agree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_gallery.models import registry

# HF key -> our Config field. Tried in order; first key present in the HF config wins.
FIELD_MAP: dict[str, list[str]] = {
    "vocab_size": ["vocab_size"],
    "context_length": ["max_position_embeddings"],
    "n_layer": ["num_hidden_layers", "n_layer"],
    "n_head": ["num_attention_heads", "n_head"],
    "n_kv_head": ["num_key_value_heads"],
    "n_embd": ["hidden_size", "n_embd"],
    "head_dim": ["head_dim"],
    "intermediate_size": ["intermediate_size"],
    "moe_intermediate_size": ["moe_intermediate_size"],
    "n_experts": ["num_experts", "n_routed_experts", "num_local_experts", "num_routed_experts"],
    "n_experts_per_tok": ["num_experts_per_tok", "num_experts_per_token", "top_k"],
    "n_shared_experts": ["n_shared_experts", "num_shared_experts"],
    "first_k_dense": ["first_k_dense_replace"],
    "q_lora_rank": ["q_lora_rank"],
    "kv_lora_rank": ["kv_lora_rank"],
    "qk_nope_head_dim": ["qk_nope_head_dim"],
    "qk_rope_head_dim": ["qk_rope_head_dim"],
    "v_head_dim": ["v_head_dim"],
    "rope_theta": ["rope_theta"],
    "norm_eps": ["rms_norm_eps", "layer_norm_epsilon", "norm_eps"],
    "tie_embeddings": ["tie_word_embeddings"],
    "sliding_window": ["sliding_window"],
    "norm_topk_prob": ["norm_topk_prob"],
    "routed_scaling_factor": ["routed_scaling_factor"],
    "query_pre_attn_scalar": ["query_pre_attn_scalar"],
    "dense_intermediate_size": ["intermediate_size"],
}

SPEC_KEY_MAP = {
    "vocab_size": "vocab",
    "context_length": "ctx",
    "n_kv_head": "n_kv",
    "intermediate_size": "inter",
    "n_experts_per_tok": "top_k",
    "moe_intermediate_size": "moe_inter",
    "n_shared_experts": "n_shared",
    "q_lora_rank": "q_lora",
    "kv_lora_rank": "kv_lora",
    "qk_nope_head_dim": "qk_nope",
    "qk_rope_head_dim": "qk_rope",
    "v_head_dim": "v_head",
    "routed_scaling_factor": "scaling",
    "rope_theta": "theta",
    "norm_eps": "eps",
    "tie_embeddings": "tie",
    "query_pre_attn_scalar": "qpa",
    "sliding_window": "window",
    "dense_intermediate_size": "dense_inter",
}

GENERATED_MODULES = {
    "llama3_2_3b",
    "mistral_small_3_1",
    "phi_4",
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_32b",
    "gemma3_270m",
    "qwen3_235b_a22b",
    "qwen3_coder_flash",
    "glm_4_5",
    "glm_4_7",
    "glm_4_5_air",
    "grok_2_5",
    "minimax_m2",
    "minimax_m2_5",
    "minimax_m2_7",
    "intellect_3",
    "sarvam_30b",
    "gpt_oss_120b",
    "deepseek_r1",
    "deepseek_v3_2",
    "deepseek_v4_flash",
    "deepseek_v4_pro",
    "kimi_k2",
    "kimi_k2_5",
    "kimi_k2_6",
    "kimi_k2_7_code",
    "mistral_large_3",
    "glm_5",
    "glm_5_1",
    "glm_5_2",
    "sarvam_105b",
    "longcat_flash_lite",
    "mistral_small_4",
    "nemotron3_super_120b",
    "nemotron3_ultra",
    "nemotron3_nano_4b",
    "qwen3_5",
    "qwen3_6_35b_a3b",
    "ling_2_5",
    "ling_2_6",
    "olmo3_7b",
    "olmo3_32b",
    "nanbeige_4_1",
    "granite_4_1",
    "lfm2_5_1_2b",
    "lfm2_5_350m",
    "qwen3_6_27b",
    "vibethinker_3b",
    "gemma4_31b",
    "gemma4_12b",
    "gemma4_e4b",
    "gemma4_e2b",
    "llama4_maverick",
    "arcee_trinity_large",
    "step_3_5_flash",
    "mimo_v2_flash",
    "mimo_v2_5",
    "mimo_v2_5_pro",
    "hunyuan_3_preview",
    "command_a_plus",
    "lfm2_5_8b_a1b",
    "mellum2_thinking",
    "laguna_xs_2",
    "zaya1_8b",
    "gemma4_26b_a4b",
}

# These public-config fields intentionally differ from the educational Config
# surface. Most are MoE variants where the HF ``intermediate_size`` describes a
# full/dense FFN width while the gallery module keeps a smaller active or shared
# FFN width to preserve tiny-preset runtime. A few sliding-window entries are
# documented assumptions where the model keeps schedule metadata visible without
# pretending exact production cadence.
FIELD_DIFF_ALLOWLIST: dict[str, set[str]] = {
    "command-a-plus": {"intermediate_size", "sliding_window"},
    "gemma4-26b-a4b": {"intermediate_size", "sliding_window"},
    "glm-4.5": {"intermediate_size"},
    "glm-4.5-air": {"intermediate_size"},
    "glm-4.7": {"intermediate_size"},
    "grok-2.5": {"intermediate_size"},
    "hunyuan-3-preview": {"intermediate_size"},
    "intellect-3": {"intermediate_size"},
    "laguna-xs.2": {"intermediate_size"},
    "lfm2.5-8b-a1b": {"intermediate_size"},
    "mellum2-thinking": {"intermediate_size", "sliding_window"},
    "mimo-v2-flash": {"intermediate_size"},
    "mimo-v2.5": {"intermediate_size", "sliding_window"},
    "mimo-v2.5-pro": {"intermediate_size", "sliding_window"},
    "qwen3-235b-a22b": {"intermediate_size"},
    "qwen3-coder-flash": {"intermediate_size"},
    "sarvam-30b": {"intermediate_size"},
    "step-3.5-flash": {"intermediate_size"},
}


@dataclass(frozen=True)
class FieldDiff:
    field: str
    ours: Any
    expected: Any
    source: str


@dataclass(frozen=True)
class AuditResult:
    slug: str
    module: str
    preset_name: str
    gallery_scale: str | None
    param_count: int | None
    published_params: float | None
    param_error: str | None
    field_diffs: tuple[FieldDiff, ...]
    has_public_config: bool

    @property
    def has_findings(self) -> bool:
        return bool(self.field_diffs or self.param_error or self.param_count_diff)

    @property
    def param_count_diff(self) -> str | None:
        if self.param_count is None or self.published_params is None:
            return None
        rel = (self.param_count - self.published_params) / self.published_params
        if abs(rel) <= 0.05:
            return None
        return f"params: ours {self.param_count / 1e9:.2f}B vs published {self.published_params / 1e9:.2f}B ({rel:+.1%})"


def real_preset(mod):
    """The non-tiny preset with published dims (DEFAULT_PRESET when several exist)."""
    names = [k for k in mod.PRESETS if k != "tiny"]
    name = mod.DEFAULT_PRESET if mod.DEFAULT_PRESET in names else names[-1]
    return name, mod.PRESETS[name]


def hf_text_config(raw: dict) -> dict:
    return raw.get("text_config", raw)


def parse_scale(scale: str | None) -> float | None:
    """'8B parameters' -> 8e9; '270M parameters' -> 2.7e8; '1.02T ...' -> 1.02e12."""
    m = re.search(r"([\d.]+)\s*([MBT])", scale or "")
    if not m:
        return None
    return float(m.group(1)) * {"M": 1e6, "B": 1e9, "T": 1e12}[m.group(2)]


def parse_gallery_context(card: dict[str, Any]) -> int | None:
    gal_ctx = (card.get("context") or "").replace(",", "")
    return int(gal_ctx) if gal_ctx.isdigit() else None


def count_meta_params(mod, cfg) -> int:
    with torch.device("meta"):
        model = mod.Model(cfg)
    return sum(p.numel() for p in model.parameters())


def load_cards(cards_path: str | Path, mapping_path: str | Path) -> dict[str, dict[str, Any]]:
    cards = {c["id"].removeprefix("card-"): c for c in json.loads(Path(cards_path).read_text())}
    mapping = json.loads(Path(mapping_path).read_text())
    return {slug: cards.get(card_id, {}) for slug, card_id in mapping.items()}


def load_hf_config(config_dir: str | Path, slug: str) -> dict[str, Any] | None:
    path = Path(config_dir) / f"{slug}.json"
    if not path.exists():
        return None
    return hf_text_config(json.loads(path.read_text()))


def hf_value_for_field(hf: dict[str, Any], field: str) -> Any:
    if field == "dense_intermediate_size" and "moe_intermediate_size" not in hf:
        return None
    return next((hf[k] for k in FIELD_MAP[field] if k in hf), None)


def expected_fields(
    cfg, hf: dict[str, Any] | None, card: dict[str, Any]
) -> dict[str, tuple[Any, str]]:
    field_names = {f.name for f in fields(cfg)}
    expected: dict[str, tuple[Any, str]] = {}

    if "context_length" in field_names:
        if hf and "max_position_embeddings" in hf:
            expected["context_length"] = (hf["max_position_embeddings"], "config.json")
        else:
            gallery_ctx = parse_gallery_context(card)
            if gallery_ctx is not None:
                expected["context_length"] = (gallery_ctx, "gallery")

    if not hf:
        return expected

    for field in FIELD_MAP:
        if field == "context_length" or field not in field_names:
            continue
        hf_val = hf_value_for_field(hf, field)
        if hf_val is None or isinstance(hf_val, (list, dict)):
            continue
        expected[field] = (hf_val, "config.json")
    return expected


def values_match(ours: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(ours, bool):
        return bool(expected) == bool(ours)
    if isinstance(expected, (int, float)) and isinstance(ours, (int, float)):
        return float(expected) == float(ours)
    return expected == ours


def field_diffs_for(
    slug: str,
    config_dir: str | Path,
    cards_by_slug: dict[str, dict[str, Any]] | None = None,
    *,
    include_allowed: bool = False,
) -> list[FieldDiff]:
    mod = registry.load(slug)
    _, cfg = real_preset(mod)
    hf = load_hf_config(config_dir, slug)
    card = (cards_by_slug or {}).get(slug, {})
    ours = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else vars(cfg)
    diffs: list[FieldDiff] = []
    allowed = FIELD_DIFF_ALLOWLIST.get(slug, set())
    for field, (expected, source) in expected_fields(cfg, hf, card).items():
        if not values_match(ours[field], expected) and (include_allowed or field not in allowed):
            diffs.append(FieldDiff(field, ours[field], expected, source))
    return diffs


def audit(
    *,
    cards_by_slug: dict[str, dict[str, Any]],
    config_dir: str | Path,
    include_param_counts: bool = True,
) -> list[AuditResult]:
    results: list[AuditResult] = []
    for entry in registry.REGISTRY:
        slug = entry.slug
        card = cards_by_slug.get(slug, {})
        mod = registry.load(slug)
        preset_name, cfg = real_preset(mod)
        param_count = None
        param_error = None

        if include_param_counts:
            try:
                param_count = count_meta_params(mod, cfg)
            except Exception as e:  # meta-device incompatibility etc.
                param_error = f"param-count failed: {e}"

        hf = load_hf_config(config_dir, slug)
        results.append(
            AuditResult(
                slug=slug,
                module=entry.module,
                preset_name=preset_name,
                gallery_scale=card.get("scale"),
                param_count=param_count,
                published_params=parse_scale(card.get("scale")),
                param_error=param_error,
                field_diffs=tuple(field_diffs_for(slug, config_dir, cards_by_slug)),
                has_public_config=hf is not None,
            )
        )
    return results


def spec_overrides(results: list[AuditResult]) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    entries_by_slug = {e.slug: e for e in registry.REGISTRY}
    for result in results:
        entry = entries_by_slug[result.slug]
        if entry.module not in GENERATED_MODULES:
            continue
        converted: dict[str, Any] = {}
        for diff in result.field_diffs:
            key = SPEC_KEY_MAP.get(diff.field, diff.field)
            converted[key] = diff.expected
        if converted:
            overrides[result.slug] = converted
    return overrides


def print_report(results: list[AuditResult]) -> None:
    n_diff = n_clean = n_nocfg = 0
    for result in results:
        lines: list[str] = []
        if result.param_error:
            lines.append(result.param_error)
        if result.param_count_diff:
            lines.append(result.param_count_diff)
        if not result.has_public_config:
            n_nocfg += 1
            lines.append("(no public config.json - gallery attrs only)")
        lines.extend(
            f"{d.field}: ours {d.ours!r} vs {d.source} {d.expected!r}" for d in result.field_diffs
        )
        if lines:
            n_diff += 1
            param_text = (
                f", params {result.param_count:,}" if result.param_count is not None else ""
            )
            print(
                f"\n### {result.slug}  [{result.preset_name}]  ({result.gallery_scale}{param_text})"
            )
            for line in lines:
                print(f"  - {line}")
        else:
            n_clean += 1
    print(f"\n== {n_clean} clean, {n_diff} with findings, {n_nocfg} without public config ==")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="/tmp/gallery_cards.json")
    ap.add_argument("--mapping", default="/tmp/slug_to_card.json")
    ap.add_argument("--configs", default="/tmp/hf_configs")
    ap.add_argument(
        "--no-param-counts", action="store_true", help="skip meta-device parameter counting"
    )
    ap.add_argument(
        "--emit-param-counts", action="store_true", help="print JSON {slug: real_param_count}"
    )
    ap.add_argument(
        "--emit-specs",
        action="store_true",
        help="print a machine-usable SPEC_OVERRIDES dict for generated variants",
    )
    args = ap.parse_args()

    cards_by_slug = load_cards(args.cards, args.mapping)
    results = audit(
        cards_by_slug=cards_by_slug,
        config_dir=args.configs,
        include_param_counts=not args.no_param_counts or args.emit_param_counts,
    )

    if args.emit_param_counts:
        print(json.dumps({r.slug: r.param_count for r in results}, indent=2, sort_keys=True))
        return

    if args.emit_specs:
        print("SPEC_OVERRIDES = " + repr(spec_overrides(results)))
        return

    print_report(results)


if __name__ == "__main__":
    main()
