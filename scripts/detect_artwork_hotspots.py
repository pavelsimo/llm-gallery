"""Detect dense clickable hotspot boxes from architecture artwork.

The visualizer renders Sebastian Raschka's original architecture artwork. This
helper finds the small rounded boxes in that artwork, OCRs their labels, and
maps them back to the generated code sections/anchors. It intentionally emits
only artwork-coordinate hotspots that resolve to code.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
MANIFEST_PATH = WEB_ROOT / "assets" / "architectures" / "manifest.json"
HOTSPOTS_DIR = WEB_ROOT / "assets" / "architectures" / "hotspots"
DEFAULT_SOURCE_DIR = Path("/tmp/llm-gallery-assets-probe/all")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_gallery.models import registry  # noqa: E402


CC_RE = re.compile(
    r"\s*\d+:\s+"
    r"(?P<w>\d+)x(?P<h>\d+)\+(?P<x>\d+)\+(?P<y>\d+)\s+"
    r"[^ ]+\s+(?P<area>[0-9.eE+-]+)\s+gray\((?P<color>\d+)\)"
)

MAIN_STACK_TOP_TO_BOTTOM = [
    "lm-head",
    "final-norm",
    "ffn-dropout",
    "feed-forward",
    "block-norm-2",
    "attn-dropout",
    "compute",
    "block-norm-1",
    "embed-dropout",
    "position",
    "embedding",
    "input",
]

NODE_ALIASES = {
    "input": ("tokenized text", "input tokens"),
    "embedding": ("token embedding", "token embedding layer", "embedding layer"),
    "position": (
        "positional embedding",
        "positional embedding layer",
        "position embedding",
        "rope",
        "rotary",
        "rotary embedding",
        "partial rope",
    ),
    "embed-dropout": ("dropout", "embedding dropout"),
    "block-norm-1": ("norm 1", "rmsnorm 1", "layernorm 1", "layer norm 1"),
    "compute": (
        "attention",
        "masked attention",
        "masked multi head attention",
        "multi head attention",
        "grouped query attention",
        "multi head latent attention",
        "mla",
        "linear attention",
        "gated delta rule",
        "mamba",
        "mlstm",
        "xlstm",
        "sequence mixer",
    ),
    "attn-dropout": ("dropout", "attention dropout"),
    "plus-1": ("+", "attention residual"),
    "block-norm-2": ("norm 2", "rmsnorm 2", "layernorm 2", "layer norm 2"),
    "feed-forward": (
        "feed forward",
        "feedforward",
        "ffn",
        "mlp",
        "swiglu",
        "geglu",
        "moe",
        "expert",
        "deepseekmoe",
    ),
    "ffn-dropout": ("dropout", "mlp dropout", "ffn dropout"),
    "plus-2": ("+", "feed forward residual", "mlp residual"),
    "final-norm": ("final norm", "final rmsnorm", "final layernorm", "final layer norm"),
    "lm-head": ("linear output layer", "lm head", "language model head"),
    "qk-norm": ("qk norm", "q k norm", "q/k norm", "qk normalization"),
    "ffn-gate": ("gate projection", "gate proj", "linear layer"),
    "ffn-up": ("up projection", "up proj", "linear layer"),
    "ffn-act": ("silu activation", "gelu activation", "activation"),
    "ffn-down": ("down projection", "down proj", "linear layer", "output projection"),
    "moe-router": ("router", "gate", "router gate"),
    "moe-topk": ("top k", "top-k", "topk", "expert select"),
    "moe-experts": ("expert", "experts", "expert dispatch"),
    "moe-shared": ("shared expert",),
    "mla-query": ("query low rank", "query compression", "q low rank"),
    "mla-kv": ("latent kv", "kv cache", "latent kv cache", "kv compression"),
    "mla-rope": ("decoupled rope", "rope"),
    "mla-out": ("output projection", "linear layer"),
    "mixer-proj": ("q k v gates", "qkv gates", "projection", "linear"),
    "mixer-state": ("state update", "gated delta rule", "mlstm", "mamba"),
    "mixer-out": ("read output", "output projection", "linear"),
}


@dataclass
class Component:
    x: int
    y: int
    w: int
    h: int
    area: float
    color: int
    density: float
    texts: list[str] = field(default_factory=list)
    target_id: str | None = None
    label: str = ""
    role: str = ""
    source: str = "detected"

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def manifest_assets() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["assets"]


def image_path(slug: str, assets: dict[str, Any], source_dir: Path) -> Path:
    filename = Path(assets[slug]["source_url"]).name
    path = source_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True)


def connected_components(path: Path) -> list[Component]:
    result = run_command(
        [
            "magick",
            str(path),
            "-colorspace",
            "Gray",
            "-threshold",
            "35%",
            "-define",
            "connected-components:verbose=true",
            "-connected-components",
            "4",
            "null:",
        ]
    )
    components: list[Component] = []
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        match = CC_RE.match(line)
        if not match:
            continue
        width = int(match.group("w"))
        height = int(match.group("h"))
        area = float(match.group("area"))
        density = area / (width * height)
        components.append(
            Component(
                x=int(match.group("x")),
                y=int(match.group("y")),
                w=width,
                h=height,
                area=area,
                color=int(match.group("color")),
                density=density,
            )
        )
    return components


def expand_component(component: Component, image_width: int, image_height: int, pad: int = 10) -> Component:
    x = max(0, component.x - pad)
    y = max(0, component.y - pad)
    x2 = min(image_width, component.x2 + pad)
    y2 = min(image_height, component.y2 + pad)
    return Component(
        x=x,
        y=y,
        w=x2 - x,
        h=y2 - y,
        area=component.area,
        color=component.color,
        density=component.density,
        texts=component.texts,
    )


def is_small_box(component: Component, image_width: int, image_height: int) -> bool:
    if component.y < 150:
        return False
    if component.w < 68 or component.h < 25:
        return False
    if component.w > min(900, int(image_width * 0.42)):
        return False
    if component.h > min(340, int(image_height * 0.18)):
        return False
    if component.area < 900 or component.density < 0.45:
        return False
    aspect = component.w / component.h
    return 0.45 <= aspect <= 9.5


def find_model_shell(components: list[Component], image_width: int, image_height: int) -> Component | None:
    candidates = [
        component
        for component in components
        if component.color == 255
        and component.density > 0.35
        and image_width * 0.25 <= component.w <= image_width * 0.7
        and image_height * 0.35 <= component.h <= image_height * 0.9
        and component.x <= image_width * 0.55
        and component.y > image_height * 0.04
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda component: component.area)


def inside(component: Component, bounds: Component, margin: int = 0) -> bool:
    return (
        component.x >= bounds.x + margin
        and component.y >= bounds.y + margin
        and component.x2 <= bounds.x2 - margin
        and component.y2 <= bounds.y2 - margin
    )


def is_main_stack_box(component: Component, model_shell: Component | None) -> bool:
    if model_shell is None or not inside(component, model_shell, margin=-14):
        return False
    left = model_shell.x + model_shell.w * 0.18
    right = model_shell.x + model_shell.w * 0.82
    return left <= component.cx <= right


def crop_ocr(image: Path, component: Component, image_width: int, image_height: int) -> list[str]:
    pad = 8
    x = max(0, component.x - pad)
    y = max(0, component.y - pad)
    x2 = min(image_width, component.x2 + pad)
    y2 = min(image_height, component.y2 + pad)
    crop = f"{x2 - x}x{y2 - y}+{x}+{y}"

    with tempfile.TemporaryDirectory(prefix="llm-gallery-ocr-") as tmp:
        gray = Path(tmp) / "crop-gray.png"
        bw = Path(tmp) / "crop-bw.png"
        subprocess.run(
            [
                "magick",
                str(image),
                "-crop",
                crop,
                "+repage",
                "-resize",
                "230%",
                "-colorspace",
                "Gray",
                str(gray),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["magick", str(gray), "-threshold", "72%", str(bw)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        texts: list[str] = []
        for path, psm in ((bw, "6"), (bw, "13"), (gray, "6")):
            if texts and useful_text(texts[0]):
                break
            result = subprocess.run(
                ["tesseract", str(path), "stdout", "--psm", psm],
                check=False,
                capture_output=True,
                text=True,
            )
            text = clean_text(result.stdout)
            if text and text not in texts:
                texts.append(text)
        return texts


def clean_text(value: str) -> str:
    value = " ".join(value.split())
    value = value.strip(" _-—–|[](){}'\"“”‘’.,:;\\/`~")
    value = value.replace("Iinear", "Linear").replace("Iayer", "layer")
    value = value.replace("LinearA layer", "Linear layer").replace("LineaArlayer", "Linear layer")
    value = value.replace("RMSorm", "RMSNorm").replace("MSNorm", "RMSNorm")
    value = value.replace("R0PE", "RoPE").replace("ROPE", "RoPE")
    return " ".join(value.split())


def normalized(value: str) -> str:
    value = value.lower()
    replacements = {
        "0": "o",
        "1": "l",
        "／": "/",
        "–": "-",
        "—": "-",
    }
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    return "".join(char for char in value if char.isalnum())


def useful_text(value: str) -> bool:
    norm = normalized(value)
    if norm in {"line", "l", "i"}:
        return False
    return len(norm) >= 3 and any(char.isalpha() for char in norm)


def slugify(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "-", value).replace("_", "-").lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "hotspot"


def best_text(component: Component) -> str:
    useful = [text for text in component.texts if useful_text(text)]
    if useful:
        return max(useful, key=lambda text: len(normalized(text)))
    return component.texts[0] if component.texts else ""


def target_maps(
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    return ({section["id"]: section for section in sections}, {anchor["id"]: anchor for anchor in anchors})


def source_target_for_id(
    target_id: str | None,
    sections: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> dict[str, str] | None:
    if not target_id:
        return None
    sections_by_id, anchors_by_id = target_maps(sections, anchors)
    if target_id in anchors_by_id:
        anchor = anchors_by_id[target_id]
        section = sections_by_id[anchor["section_id"]]
        return {"type": "anchor", "section_label": section["label"], "role": anchor["role"]}
    if target_id in sections_by_id:
        return {"type": "section", "label": sections_by_id[target_id]["label"]}
    return None


def first_section(sections: list[dict[str, Any]], *roles: str) -> dict[str, Any] | None:
    role_set = set(roles)
    return next((section for section in sections if section["role"] in role_set), None)


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


def node_by_id(nodes: list[dict[str, Any]], node_id: str) -> dict[str, Any] | None:
    return next((node for node in nodes if node["id"] == node_id), None)


def node_target_id(nodes: list[dict[str, Any]], node_id: str) -> str | None:
    node = node_by_id(nodes, node_id)
    return node.get("target_id") if node else None


def alias_score(text_norm: str, alias: str) -> float:
    alias_norm = normalized(alias)
    if not text_norm or not alias_norm:
        return 0.0
    if alias_norm in text_norm or text_norm in alias_norm:
        return 1.0
    return SequenceMatcher(None, text_norm, alias_norm).ratio()


def node_aliases(node: dict[str, Any]) -> list[str]:
    aliases = [node.get("label", ""), node.get("role", "")]
    aliases.extend(NODE_ALIASES.get(node["id"], ()))
    return [alias for alias in aliases if alias]


def best_node_match(text: str, nodes: list[dict[str, Any]]) -> tuple[str | None, float]:
    text_norm = normalized(text)
    best_id: str | None = None
    best = 0.0
    for node in nodes:
        for alias in node_aliases(node):
            score = alias_score(text_norm, alias)
            if score > best:
                best = score
                best_id = node["id"]
    return best_id, best


def assign_direct_target(
    component: Component,
    payload: dict[str, Any],
    model_shell: Component | None,
) -> None:
    text = best_text(component)
    if not useful_text(text):
        return

    nodes = payload["diagram"]["nodes"]
    sections = payload["sections"]
    anchors = payload["anchors"]
    norm = normalized(text)
    is_main = is_main_stack_box(component, model_shell)

    def set_node(node_id: str, label: str | None = None) -> bool:
        target_id = node_target_id(nodes, node_id)
        if target_id is None:
            return False
        component.target_id = target_id
        component.label = label or node_by_id(nodes, node_id).get("label", text)
        component.role = node_by_id(nodes, node_id).get("role", "helper")
        component.source = "ocr"
        return True

    if "tokenized" in norm:
        set_node("input", "Tokenized text")
    elif "tokenembedding" in norm or ("embeddinglayer" in norm and "position" not in norm):
        set_node("embedding", "Token embedding")
    elif "positionalembedding" in norm or "positionembedding" in norm:
        set_node("position", "Positional embedding")
    elif "linearoutputlayer" in norm or "lmhead" in norm:
        set_node("lm-head", "Linear output layer")
    elif "final" in norm and "norm" in norm:
        set_node("final-norm", "Final norm")
    elif "norm" in norm and ("l" in norm[-2:] or "1" in text):
        set_node("block-norm-1", "Norm 1")
    elif "norm" in norm and ("2" in text or norm.endswith("z")):
        set_node("block-norm-2", "Norm 2")
    elif "qknorm" in norm or "qknormalization" in norm:
        set_node("qk-norm", "Q/K norm")
    elif "topk" in norm or "top" in norm and "select" in norm:
        set_node("moe-topk", "Top-k select")
    elif "sharedexpert" in norm:
        set_node("moe-shared", "Shared expert")
    elif "router" in norm:
        set_node("moe-router", "Router / gate")
    elif "expert" in norm and "shared" not in norm:
        set_node("moe-experts", "Experts")
    elif "gateprojection" in norm or "gateproj" in norm:
        set_node("ffn-gate", "Gate projection")
    elif "upprojection" in norm or "upproj" in norm:
        set_node("ffn-up", "Up projection")
    elif "downprojection" in norm or "downproj" in norm:
        set_node("ffn-down", "Down projection")
    elif "silu" in norm or "gelu" in norm or "activation" in norm:
        if not set_node("ffn-act", "Activation"):
            mlp = first_section(sections, "mlp") or first_section(sections, "expert")
            target = anchor_by_role(anchors, "activation", mlp) if mlp is not None else None
            if target is None:
                target = mlp
            if target is not None:
                component.target_id = target["id"]
                component.label = "Activation"
                component.role = mlp["role"] if mlp is not None else "mlp"
                component.source = "ocr"
    elif "feedforward" in norm:
        if is_main:
            set_node("feed-forward", "Feed forward")
        else:
            mlp = first_section(sections, "mlp", "expert") or first_section(sections, "moe")
            if mlp is not None:
                component.target_id = mlp["id"]
                component.label = "Feed forward"
                component.role = mlp["role"]
                component.source = "ocr"
    elif "rope" in norm or alias_score(norm, "rope") >= 0.78:
        if not set_node("mla-rope", "RoPE"):
            set_node("position", "RoPE")
    elif any(token in norm for token in ("latentkv", "kvcache", "kvcompression")):
        set_node("mla-kv", "Latent KV cache")
    elif "query" in norm and ("low" in norm or "compression" in norm):
        set_node("mla-query", "Query low-rank")
    elif "outputprojection" in norm:
        if not set_node("mla-out", "Output projection"):
            set_node("mixer-out", "Output projection")
    elif any(token in norm for token in ("attention", "attn", "mla")):
        if is_main:
            set_node("compute", text)
        else:
            attention = first_section(sections, "attention") or first_section(sections, "mixer")
            if attention is not None:
                component.target_id = attention["id"]
                component.label = text
                component.role = attention["role"]
                component.source = "ocr"
    elif any(token in norm for token in ("gateddelta", "mlstm", "mamba", "stateupdate")):
        if not set_node("mixer-state", text):
            mixer = first_section(sections, "mixer")
            if mixer is not None:
                component.target_id = mixer["id"]
                component.label = text
                component.role = "mixer"
                component.source = "ocr"

    if component.target_id:
        return

    ambiguous = (
        norm in {"dropout", "linear", "linearlayer", "rmsnorm", "layernorm", "norm"}
        or ("linearlayer" in norm and "output" not in norm)
    )
    if not ambiguous:
        node_id, score = best_node_match(text, nodes)
        if node_id and score >= 0.82:
            set_node(node_id, text)


def assign_main_stack_targets(
    components: list[Component],
    payload: dict[str, Any],
    model_shell: Component | None,
    slug: str,
) -> None:
    nodes = payload["diagram"]["nodes"]
    main = [
        component
        for component in components
        if component.target_id is None and is_main_stack_box(component, model_shell)
    ]
    if not main:
        return

    assigned_target_ids = {component.target_id for component in components if component.target_id}
    expected: list[dict[str, Any]] = []
    for node_id in MAIN_STACK_TOP_TO_BOTTOM:
        node = node_by_id(nodes, node_id)
        if node is None or not node.get("target_id"):
            continue
        if node_id == "position" and slug != "gpt2-xl" and node.get("x", 0) < 150:
            continue
        if node["target_id"] in assigned_target_ids and node_id not in {
            "embed-dropout",
            "attn-dropout",
            "ffn-dropout",
        }:
            continue
        expected.append(node)

    for component, node in zip(sorted(main, key=lambda item: item.y), expected, strict=False):
        component.target_id = node["target_id"]
        component.label = node.get("label") or best_text(component) or node["id"]
        component.role = node.get("role", "helper")
        component.source = "detected"


def assign_detail_fallbacks(
    components: list[Component],
    payload: dict[str, Any],
    model_shell: Component | None,
) -> None:
    sections = payload["sections"]
    anchors = payload["anchors"]
    mlp = first_section(sections, "mlp") or first_section(sections, "expert")
    moe = first_section(sections, "moe")
    mixer = first_section(sections, "mixer")
    attention = first_section(sections, "attention")

    activation_refs = [
        component
        for component in components
        if component.target_id and component.label and "activation" in normalized(component.label)
    ]
    activation_mid = (
        sum(component.cy for component in activation_refs) / len(activation_refs)
        if activation_refs
        else None
    )

    for component in components:
        if component.target_id is not None:
            continue
        text = best_text(component)
        norm = normalized(text)
        if not useful_text(text) and not (component.w <= 90 and component.h <= 90):
            continue

        target: dict[str, Any] | None = None
        label = text or "Architecture block"
        role = "helper"

        if "dropout" in norm:
            section = mlp or attention or first_section(sections, "model")
            if section is not None:
                target = anchor_by_role(anchors, "mlp_dropout", section) or section
                label = "Dropout"
                role = "dropout"
        elif "linear" in norm:
            if activation_mid is not None and mlp is not None:
                nearest_activation = min(
                    activation_refs,
                    key=lambda item: abs(item.cy - component.cy) + abs(item.cx - component.cx) * 0.25,
                )
                same_row = abs(component.cy - nearest_activation.cy) <= max(42, component.h * 0.55)
                if same_row and component.cx > nearest_activation.cx:
                    anchor_role = "mlp_gate"
                else:
                    anchor_role = "mlp_output" if component.cy < nearest_activation.cy else "mlp_gate"
                target = anchor_by_role(anchors, anchor_role, mlp) or mlp
                role = mlp["role"]
            elif mixer is not None and not is_main_stack_box(component, model_shell):
                target = anchor_by_role(anchors, "output_projection", mixer) or mixer
                role = "mixer"
            elif attention is not None and not is_main_stack_box(component, model_shell):
                target = anchor_by_role(anchors, "output_projection", attention) or attention
                role = "attention"
            else:
                target = mlp or moe or attention or mixer
                role = target["role"] if target else "helper"
            label = clean_text(text) or "Linear layer"
        elif "norm" in norm:
            target = anchor_by_role(anchors, "qk_norm", attention) or first_section(sections, "norm")
            role = "norm"
        elif any(token in norm for token in ("conv", "state", "delta", "mlstm", "mamba")):
            target = mixer or attention
            role = target["role"] if target else "mixer"
        elif any(token in norm for token in ("feedforward", "expert", "moe")):
            target = moe or mlp
            role = target["role"] if target else "mlp"
        elif any(token in norm for token in ("attention", "attn")):
            target = attention or mixer
            role = target["role"] if target else "attention"
        elif is_main_stack_box(component, model_shell):
            target = first_section(sections, "block")
            role = "block"

        if target is not None:
            component.target_id = target["id"]
            component.label = label
            component.role = role
            component.source = "ocr" if useful_text(text) else "detected"


def hotspot_source(slug: str, source_dir: Path, assets: dict[str, Any]) -> dict[str, Any]:
    from scripts import build_visualizer_data as builder

    payload = builder.build_model_payload(registry.get(slug), assets)
    artwork = payload["diagram"]["artwork"]
    source_image = image_path(slug, assets, source_dir)
    components = connected_components(source_image)
    model_shell = find_model_shell(components, artwork["width"], artwork["height"])
    candidates = [
        expand_component(component, artwork["width"], artwork["height"])
        for component in components
        if is_small_box(component, artwork["width"], artwork["height"])
    ]
    candidates.sort(key=lambda component: (component.y, component.x, component.w, component.h))

    for component in candidates:
        component.texts = crop_ocr(source_image, component, artwork["width"], artwork["height"])
        assign_direct_target(component, payload, model_shell)

    assign_main_stack_targets(candidates, payload, model_shell, slug)
    assign_detail_fallbacks(candidates, payload, model_shell)

    hotspots: list[dict[str, Any]] = []
    used_ids: dict[str, int] = {}
    for component in candidates:
        target = source_target_for_id(component.target_id, payload["sections"], payload["anchors"])
        if target is None:
            continue
        label = component.label or best_text(component) or component.target_id or "Architecture block"
        label_norm = normalized(label)
        if "norm" in label_norm and component.h > 180:
            continue
        if label_norm in {"line", "l", "i"}:
            continue
        base_id = slugify(f"{component.target_id}.{label}")
        count = used_ids.get(base_id, 0) + 1
        used_ids[base_id] = count
        hotspot_id = base_id if count == 1 else f"{base_id}-{count}"
        radius = max(8, min(80, int(min(component.w, component.h) * 0.18)))
        hotspots.append(
            {
                "id": hotspot_id,
                "label": clean_text(label),
                "role": component.role or "helper",
                "target": target,
                "shape": "roundrect",
                "x": component.x,
                "y": component.y,
                "w": component.w,
                "h": component.h,
                "rx": radius,
                "ry": radius,
                "source": component.source,
            }
        )

    hotspots.sort(key=lambda item: (item["y"], item["x"], item["id"]))
    return {
        "slug": slug,
        "coordinate_space": "artwork",
        "image_sha256": assets[slug]["source_sha256"],
        "checked": True,
        "hotspots": hotspots,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--slug", action="append", choices=[entry.slug for entry in registry.REGISTRY])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    if shutil.which("magick") is None:
        print("magick is not installed or not on PATH", file=sys.stderr)
        return 1
    if shutil.which("tesseract") is None:
        print("tesseract is not installed or not on PATH", file=sys.stderr)
        return 1

    assets = manifest_assets()
    slugs = args.slug or [entry.slug for entry in registry.REGISTRY]
    for slug in slugs:
        source = hotspot_source(slug, args.source_dir, assets)
        print(f"{slug}: {len(source['hotspots'])} artwork hotspots")
        if args.write:
            write_json(HOTSPOTS_DIR / f"{slug}.json", source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
