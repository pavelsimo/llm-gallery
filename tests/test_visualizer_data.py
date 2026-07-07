from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

from llm_gallery.models import registry

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_visualizer_data.py"
EMBEDDED_WEBP_RE = re.compile(r'href="data:image/webp;base64,([^"]+)"')

spec = importlib.util.spec_from_file_location("build_visualizer_data", BUILDER_PATH)
assert spec is not None
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)


@pytest.fixture(scope="session")
def payloads():
    return builder.build_payloads()


def test_index_lists_every_registry_entry(payloads):
    index = payloads["index.json"]
    assert index["model_count"] == len(registry.REGISTRY)
    assert [model["slug"] for model in index["models"]] == [entry.slug for entry in registry.REGISTRY]
    assert all(model["gallery_card_id"] for model in index["models"])
    assert index["data_version"]
    assert "learning_path" not in index


def test_payload_data_versions_match_index(payloads):
    version = payloads["index.json"]["data_version"]
    for filename, payload in payloads.items():
        if filename == "index.json":
            continue
        assert payload["data_version"] == version


def test_palette_tables_match_registry():
    registry_slugs = {entry.slug for entry in registry.REGISTRY}
    assert set(builder.DIAGRAM_PALETTES) == registry_slugs


def test_architecture_asset_manifest_covers_registry():
    manifest_path = ROOT / "web" / "assets" / "architectures" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest["assets"]

    assert manifest["asset_count"] == len(registry.REGISTRY) == 83
    assert list(assets) == [entry.slug for entry in registry.REGISTRY]

    for entry in registry.REGISTRY:
        asset = assets[entry.slug]
        svg_path = ROOT / "web" / asset["path"]
        assert svg_path.exists()
        assert asset["source_url"].startswith(
            "https://sebastianraschka.com/llm-architecture-gallery/images/architectures/"
        )
        assert asset["source_url"].endswith(".webp")
        assert asset["width"] > 0
        assert asset["height"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", asset["source_sha256"])
        match = EMBEDDED_WEBP_RE.search(svg_path.read_text(encoding="utf-8"))
        assert match, f"{svg_path.relative_to(ROOT)} does not embed a WebP payload"
        embedded = base64.b64decode(match.group(1), validate=True)
        assert hashlib.sha256(embedded).hexdigest() == asset["source_sha256"]
        assert "cleaned_sha256" not in asset
        assert "watermark_masks" not in asset


@pytest.mark.parametrize("slug", ["gpt2-xl", "deepseek-v3", "qwen3-30b-a3b", "kimi-linear", "xlstm-7b"])
def test_representative_svg_assets_render(slug):
    manifest = json.loads((ROOT / "web" / "assets" / "architectures" / "manifest.json").read_text(encoding="utf-8"))
    asset = manifest["assets"][slug]
    svg_path = ROOT / "web" / asset["path"]
    result = subprocess.run(
        ["magick", str(svg_path), "-format", "%w %h", "info:"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert result.stdout == f"{asset['width']} {asset['height']}"


@pytest.mark.parametrize("entry", registry.REGISTRY, ids=lambda entry: entry.slug)
def test_diagram_palette_is_valid(entry, payloads):
    palette = payloads[f"{entry.slug}.json"]["diagram"]["palette"]
    assert palette == builder.DIAGRAM_PALETTES[entry.slug]
    assert set(palette) == {"accent", "accentFill"}
    assert re.fullmatch(r"#[0-9a-f]{6}", palette["accent"])
    assert re.fullmatch(r"#[0-9a-f]{6}", palette["accentFill"])


def test_reference_palettes_match_source_screenshots(payloads):
    llama = payloads["llama3.2-1b.json"]["diagram"]["palette"]
    deepseek_v3 = payloads["deepseek-v3.json"]["diagram"]["palette"]
    deepseek_r1 = payloads["deepseek-r1.json"]["diagram"]["palette"]

    assert llama == {"accent": "#d01070", "accentFill": "#d8b0c0"}
    assert deepseek_v3 == {"accent": "#f86050", "accentFill": "#f09088"}
    assert deepseek_r1 == deepseek_v3


def test_payload_exposes_gallery_card_id_and_marked_notes(payloads):
    gpt2 = payloads["gpt2-xl.json"]
    assert gpt2["gallery_card_id"] == "gpt-2-xl-1-5b"

    deepseek = payloads["deepseek-v3.2.json"]
    assert deepseek["gallery_card_id"] == "deepseek-v3-2"
    assert any(
        note["kind"] == "assumption"
        and note["text"].startswith("sparse key-selection/index sharing is documented")
        for note in deepseek["notes"]
    )

    granite = payloads["granite-4.1.json"]
    assert any(note["kind"] == "note" and "Mamba/transformer hybrid" in note["text"] for note in granite["notes"])


def test_wave2_assumption_notes_are_exposed(payloads):
    expected = {
        "glm-5.1": "sparse key selection / index sharing is documented",
        "deepseek-v4-flash": "compressed-sparse key selection is documented",
        "mimo-v2-flash": "exact global cadence is not explicit",
        "arcee-trinity-large": "exact sliding/global cadence",
        "lfm2.5-8b-a1b": "short-conv + attention/LIV-style mixers",
        "gemma4-26b-a4b": "modeled with the generic MoE",
        "nemotron3-super-120b": "latent-MoE compression simplified",
        "ling-2.6": "full-attention layers use GQA here, not MLA",
        "kimi-linear": "the real Kimi Linear uses MLA",
    }
    for slug, needle in expected.items():
        assert any(needle in note["text"] for note in payloads[f"{slug}.json"]["notes"]), slug


@pytest.mark.parametrize("entry", registry.REGISTRY, ids=lambda entry: entry.slug)
def test_model_payload_has_valid_sections(entry, payloads):
    payload = payloads[f"{entry.slug}.json"]
    sections = payload["sections"]
    anchors = payload["anchors"]
    roles = {section["role"] for section in sections}

    assert {"config", "block", "model"} <= roles
    assert roles & builder.COMPUTE_ROLES
    assert payload["diagram"]["nodes"]
    assert payload["diagram"]["groups"]
    assert payload["diagram"]["edges"]
    assert payload["diagram"]["annotations"]
    assert payload["diagram"]["artwork"]

    line_count = len(payload["source_lines"])
    seen_ids = set()
    for section in sections:
        assert section["id"] not in seen_ids
        seen_ids.add(section["id"])
        assert 1 <= section["line_start"] <= section["line_end"] <= line_count
        assert section["line_count"] == section["line_end"] - section["line_start"] + 1

    section_ids = {section["id"] for section in sections}
    anchor_ids = set()
    for anchor in anchors:
        assert anchor["id"] not in seen_ids
        assert anchor["id"] not in anchor_ids
        anchor_ids.add(anchor["id"])
        assert anchor["section_id"] in section_ids
        assert 1 <= anchor["line_start"] <= anchor["line_end"] <= line_count
        assert anchor["line_count"] == anchor["line_end"] - anchor["line_start"] + 1
        parent = next(section for section in sections if section["id"] == anchor["section_id"])
        assert parent["line_start"] <= anchor["line_start"] <= anchor["line_end"] <= parent["line_end"]

    target_ids = section_ids | anchor_ids
    generated_target_count = 0
    for collection_name in ("groups", "nodes", "edges", "annotations"):
        for item in payload["diagram"][collection_name]:
            if item["section_id"] is not None:
                assert item["section_id"] in section_ids
            if item["target_id"] is not None:
                assert item["target_id"] in target_ids
                if collection_name in {"groups", "nodes"}:
                    generated_target_count += 1
    assert generated_target_count > 0

    artwork = payload["diagram"]["artwork"]
    assert (ROOT / "web" / artwork["path"]).exists()
    assert artwork["width"] > 0
    assert artwork["height"] > 0
    hotspots = payload["diagram"].get("hotspots", [])
    if hotspots:
        assert payload["diagram"].get("hotspotsCoordinateSpace") == "artwork"
    else:
        assert "hotspotsCoordinateSpace" not in payload["diagram"]
    for hotspot in hotspots:
        assert hotspot["target_id"] in target_ids
        assert hotspot["section_id"] is None or hotspot["section_id"] in section_ids
        assert hotspot["source"] in {"manual", "ocr", "detected"}
        assert hotspot["shape"] in {"rect", "roundrect"}
        assert 0 <= hotspot["x"] < artwork["width"]
        assert 0 <= hotspot["y"] < artwork["height"]
        assert 1 <= hotspot["w"] <= artwork["width"] - hotspot["x"]
        assert 1 <= hotspot["h"] <= artwork["height"] - hotspot["y"]


def test_hotspot_source_files_cover_registry(payloads):
    hotspot_dir = ROOT / "web" / "assets" / "architectures" / "hotspots"
    expected_paths = {hotspot_dir / f"{entry.slug}.json" for entry in registry.REGISTRY}
    actual_paths = set(hotspot_dir.glob("*.json"))
    assert actual_paths == expected_paths

    for entry in registry.REGISTRY:
        payload = payloads[f"{entry.slug}.json"]
        source_path = hotspot_dir / f"{entry.slug}.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        artwork = payload["diagram"]["artwork"]

        assert source["slug"] == entry.slug
        assert source["coordinate_space"] == "artwork"
        assert source["checked"] is True
        assert source["image_sha256"] == artwork["source_sha256"]
        assert isinstance(source["hotspots"], list)
        assert source["hotspots"], f"{source_path.relative_to(ROOT)} has no reviewed hotspots"
        assert payload["diagram"].get("hotspots")
        assert payload["diagram"].get("hotspotsCoordinateSpace") == "artwork"

        seen_ids = set()
        for hotspot in source["hotspots"]:
            assert hotspot["id"] not in seen_ids
            seen_ids.add(hotspot["id"])
            assert hotspot.get("source") in {"manual", "ocr", "detected", "generated-template"}
            if source["checked"]:
                assert hotspot["source"] != "generated-template"
            assert hotspot.get("role")
            assert hotspot.get("label")
            assert hotspot.get("shape", "rect") in {"rect", "roundrect"}
            if hotspot.get("shape") == "roundrect":
                assert isinstance(hotspot.get("rx", 0), int)
                assert isinstance(hotspot.get("ry", hotspot.get("rx", 0)), int)
            target = builder.resolve_hotspot_target(
                entry.slug,
                hotspot,
                payload["sections"],
                payload["anchors"],
            )
            assert target["id"]
            assert 0 <= hotspot["x"] < artwork["width"]
            assert 0 <= hotspot["y"] < artwork["height"]
            assert 1 <= hotspot["w"] <= artwork["width"] - hotspot["x"]
            assert 1 <= hotspot["h"] <= artwork["height"] - hotspot["y"]


def test_unchecked_generated_template_hotspots_are_not_emitted(payloads):
    hotspot_dir = ROOT / "web" / "assets" / "architectures" / "hotspots"
    for entry in registry.REGISTRY:
        payload = payloads[f"{entry.slug}.json"]
        source = json.loads((hotspot_dir / f"{entry.slug}.json").read_text(encoding="utf-8"))
        if source["checked"]:
            continue
        assert any(hotspot.get("source") == "generated-template" for hotspot in source["hotspots"])
        assert not payload["diagram"].get("hotspots")
        assert "hotspotsCoordinateSpace" not in payload["diagram"]


def test_checked_generated_template_hotspots_are_rejected(payloads, tmp_path, monkeypatch):
    slug = "qwen3-30b-a3b"
    payload = payloads[f"{slug}.json"]
    artwork = payload["diagram"]["artwork"]
    source_dir = tmp_path / "hotspots"
    source_dir.mkdir()
    (source_dir / f"{slug}.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "coordinate_space": "artwork",
                "image_sha256": artwork["source_sha256"],
                "checked": True,
                "hotspots": [
                    {
                        "id": "bad-template",
                        "label": "Attention",
                        "role": "attention",
                        "target": {"type": "section", "label": "Attention"},
                        "shape": "rect",
                        "x": 0,
                        "y": 0,
                        "w": 10,
                        "h": 10,
                        "source": "generated-template",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "HOTSPOTS_DIR", source_dir)
    with pytest.raises(ValueError, match="generated-template"):
        builder.add_hotspot_metadata(
            slug,
            {"artwork": artwork.copy()},
            payload["sections"],
            payload["anchors"],
        )


@pytest.mark.parametrize("entry", registry.REGISTRY, ids=lambda entry: entry.slug)
def test_diagram_schema_matches_template(entry, payloads):
    payload = payloads[f"{entry.slug}.json"]
    diagram = payload["diagram"]
    node_roles = {node["role"] for node in diagram["nodes"]}
    group_ids = {group["id"] for group in diagram["groups"]}
    anchor_roles = {anchor["role"] for anchor in payload["anchors"]}

    assert {"model-shell", "block-shell"} <= group_ids
    assert any(edge["role"] == "residual" for edge in diagram["edges"])
    assert any(node["role"] == "badge" for node in diagram["nodes"] if "badge" in node["id"]) or diagram["groups"]

    if payload["template"] == "dense":
        assert {"attention", "mlp"} <= node_roles
        assert "ffn-detail" in group_ids
    elif payload["template"] == "moe":
        assert {"router", "expert"} <= node_roles
        assert "moe-detail" in group_ids
    elif payload["template"] == "mla":
        assert {"attention", "latent"} <= node_roles
        assert "mla-detail" in group_ids
    elif payload["template"] == "hybrid":
        assert "mixer" in node_roles
        assert "mixer-detail" in group_ids
    elif payload["template"] == "parallel":
        assert "parallel" in anchor_roles
        assert "attention" in node_roles
        assert "mlp" in node_roles or "moe" in node_roles


@pytest.mark.parametrize("entry", registry.REGISTRY, ids=lambda entry: entry.slug)
def test_config_callouts_are_emitted_when_fields_exist(entry, payloads):
    payload = payloads[f"{entry.slug}.json"]
    labels = {annotation["label"] for annotation in payload["diagram"]["annotations"]}
    decorations = {decoration["id"] for decoration in payload["diagram"].get("decorations", [])}
    config = payload["config"]

    if "vocab_size" in config:
        assert any(label.startswith("Vocabulary size of ") for label in labels)
    if "context_length" in config:
        assert any(label.startswith("Supported context length") for label in labels)
    if "n_layer" in config:
        assert "repeat-count" in decorations
        assert "repeat-brace" in decorations
    if "n_head" in config or "linear_n_head" in config:
        assert any("heads" in label for label in labels)
    if "n_embd" in config:
        assert any(label.startswith("Embedding dimension of ") for label in labels)
    if builder.ffn_hidden_dimension(config) is not None:
        assert any(label.startswith("Hidden layer dimension of ") for label in labels)


def test_gpt2_diagram_matches_classic_reference(payloads):
    payload = payloads["gpt2-xl.json"]
    diagram = payload["diagram"]
    nodes = {node["id"]: node for node in diagram["nodes"]}
    labels = {node["label"] for node in diagram["nodes"]}
    annotations = {annotation["id"]: annotation for annotation in diagram["annotations"]}
    decorations = {decoration["id"]: decoration for decoration in diagram.get("decorations", [])}
    groups = {group["id"]: group for group in diagram["groups"]}

    assert diagram["profile"] == "gpt2"
    assert {
        "Token embedding layer",
        "Positional embedding layer",
        "Dropout",
        "LayerNorm 1",
        "Masked multi-head attention",
        "LayerNorm 2",
        "Feed forward",
        "Final LayerNorm",
        "Linear output layer",
        "Linear layer",
        "GELU activation",
    } <= labels
    assert "SwiGLU / MLP detail" not in {group["label"] for group in diagram["groups"]}
    assert "Gate projection" not in labels
    assert "Up projection" not in labels
    assert "metric-context" not in annotations
    assert "metric-layers" not in annotations
    assert annotations["metric-vocab"]["label"] == "Vocabulary size of 50,257"
    assert annotations["metric-heads"]["label"] == "25 heads"
    assert (
        annotations["metric-context-position"]["label"]
        == "Supported context length of 1,024 tokens with absolute position embeddings"
    )
    assert annotations["metric-context-position"]["target_id"] == "model.position-embedding"
    assert annotations["metric-context-token-input"]["label"] == "Supported context length of 1024 tokens"
    assert annotations["metric-context-token-input"]["target_id"] == "model.embedding"
    assert annotations["metric-embedding"]["label"] == "Embedding dimension of 1,600"
    assert annotations["metric-embedding"]["target_id"] == "model.embedding"
    assert annotations["metric-hidden"]["label"] == "Hidden layer dimension of 6,400"
    assert annotations["metric-hidden"]["target_id"] == "mlp"
    assert nodes["compute"]["tone"] == "dark"
    assert nodes["input"]["subtitle"] == "Every effort moves you"
    assert decorations["repeat-brace"]["path"]
    assert decorations["repeat-count"]["lines"][0][0]["text"] == "48 \u00d7"
    assert groups["block-shell"]["showLabel"] is False
    assert groups["ffn-detail"]["showLabel"] is False
    assert groups["ffn-detail"]["outline"] == "dotted"


@pytest.mark.parametrize("entry", registry.REGISTRY, ids=lambda entry: entry.slug)
def test_source_lines_round_trip(entry, payloads):
    payload = payloads[f"{entry.slug}.json"]
    source_path = ROOT / payload["source_path"]
    assert payload["source_lines"] == source_path.read_text(encoding="utf-8").splitlines()
    assert len(payload["source_tokens"]) == len(payload["source_lines"])
    for source_line, token_line in zip(payload["source_lines"], payload["source_tokens"], strict=True):
        assert "".join(token["t"] for token in token_line) == source_line


def test_build_time_highlighting_marks_python_tokens(payloads):
    tokens = [token for line in payloads["gpt2-xl.json"]["source_tokens"] for token in line]
    assert any(token == {"t": "class", "c": "py-keyword"} for token in tokens)
    assert any(token.get("t") == "Config" and token.get("c") == "py-class" for token in tokens)
    assert any(token.get("c") == "py-comment" for token in tokens)


def test_committed_visualizer_data_is_current(payloads):
    for filename, payload in payloads.items():
        path = ROOT / "web" / "data" / filename
        assert path.exists(), f"missing {path.relative_to(ROOT)}"
        assert path.read_text(encoding="utf-8") == builder.json_text(payload), (
            f"{path.relative_to(ROOT)} is stale; run "
            "uv run python scripts/build_visualizer_data.py"
        )
