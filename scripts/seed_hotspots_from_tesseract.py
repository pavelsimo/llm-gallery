"""Seed artwork hotspot JSON files from Tesseract OCR output.

This is a review helper, not part of the normal build. It scans architecture
artwork, fuzzy-matches OCR text against extracted model sections, and writes
unchecked hotspot source files that can be refined in ``web/hotspot-editor.html``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
DATA_DIR = WEB_ROOT / "data"
MANIFEST_PATH = WEB_ROOT / "assets" / "architectures" / "manifest.json"
HOTSPOTS_DIR = WEB_ROOT / "assets" / "architectures" / "hotspots"
DEFAULT_SOURCE_DIR = Path("/tmp/llm-gallery-assets-probe/all")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_gallery.models import registry  # noqa: E402


@dataclass(frozen=True)
class OcrBox:
    text: str
    confidence: float
    x: int
    y: int
    w: int
    h: int


def manifest_assets() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["assets"]


def payload(slug: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / f"{slug}.json").read_text(encoding="utf-8"))


def image_path(slug: str, source_dir: Path, assets: dict[str, Any]) -> Path:
    filename = Path(assets[slug]["source_url"]).name
    path = source_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def tesseract_tsv(path: Path) -> list[OcrBox]:
    result = subprocess.run(
        ["tesseract", str(path), "stdout", "tsv"],
        check=True,
        capture_output=True,
        text=True,
    )
    boxes: list[OcrBox] = []
    reader = csv.DictReader(result.stdout.splitlines(), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf") or "-1")
            x = int(row.get("left") or "0")
            y = int(row.get("top") or "0")
            width = int(row.get("width") or "0")
            height = int(row.get("height") or "0")
        except ValueError:
            continue
        if confidence < 0 or width <= 0 or height <= 0:
            continue
        boxes.append(OcrBox(text=text, confidence=confidence, x=x, y=y, w=width, h=height))
    return boxes


def normalized(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def score_text(needle: str, candidate: str) -> float:
    left = normalized(needle)
    right = normalized(candidate)
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def section_candidates(model: dict[str, Any]) -> list[dict[str, Any]]:
    roles = {"model", "block", "attention", "mixer", "mlp", "moe", "expert"}
    return [section for section in model["sections"] if section["role"] in roles]


def expand_box(boxes: list[OcrBox], pad: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(box.x for box in boxes) - pad)
    y1 = max(0, min(box.y for box in boxes) - pad)
    x2 = min(image_width, max(box.x + box.w for box in boxes) + pad)
    y2 = min(image_height, max(box.y + box.h for box in boxes) + pad)
    return x1, y1, x2 - x1, y2 - y1


def seed_for_model(slug: str, source_dir: Path, min_score: float) -> dict[str, Any]:
    assets = manifest_assets()
    model = payload(slug)
    artwork = model["diagram"]["artwork"]
    boxes = tesseract_tsv(image_path(slug, source_dir, assets))
    hotspots: list[dict[str, Any]] = []
    used_labels: set[str] = set()

    for section in section_candidates(model):
        matches = [
            box
            for box in boxes
            if score_text(section["label"], box.text) >= min_score
            or score_text(section["role"], box.text) >= min_score
        ]
        if not matches or section["label"] in used_labels:
            continue
        x, y, width, height = expand_box(matches[:4], 24, artwork["width"], artwork["height"])
        hotspots.append(
            {
                "id": f"{section['role']}-{len(hotspots) + 1}",
                "label": section["label"],
                "role": section["role"],
                "target": {"type": "section", "label": section["label"]},
                "shape": "rect",
                "x": x,
                "y": y,
                "w": width,
                "h": height,
                "source": "ocr",
            }
        )
        used_labels.add(section["label"])

    return {
        "slug": slug,
        "coordinate_space": "artwork",
        "image_sha256": artwork["source_sha256"],
        "checked": False,
        "hotspots": hotspots,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--slug", action="append", choices=[entry.slug for entry in registry.REGISTRY])
    parser.add_argument("--min-score", type=float, default=0.82)
    parser.add_argument("--write", action="store_true", help="write hotspot JSON files")
    args = parser.parse_args(argv)

    if shutil.which("tesseract") is None:
        print("tesseract is not installed or not on PATH", file=sys.stderr)
        return 1
    slugs = args.slug or [entry.slug for entry in registry.REGISTRY]
    for slug in slugs:
        seeded = seed_for_model(slug, args.source_dir, args.min_score)
        print(f"{slug}: {len(seeded['hotspots'])} OCR hotspot candidates")
        if args.write:
            write_json(HOTSPOTS_DIR / f"{slug}.json", seeded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
