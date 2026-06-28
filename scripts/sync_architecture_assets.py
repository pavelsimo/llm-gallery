"""Fetch gallery architecture images and build self-contained SVG assets.

The generated SVGs embed the live gallery artwork without modifying pixels.
They are intentionally raster-in-SVG assets: the browser overlays separate
interactive hotspots from ``scripts/build_visualizer_data.py``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
ASSET_DIR = WEB_ROOT / "assets" / "architectures"
MANIFEST_PATH = ASSET_DIR / "manifest.json"
CACHE_DIR = ROOT / ".cache" / "llm-gallery" / "architectures"
GALLERY_URL = "https://sebastianraschka.com/llm-architecture-gallery"
EMBEDDED_WEBP_RE = re.compile(r'href="data:image/webp;base64,([^"]+)"')

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_gallery.models import registry  # noqa: E402,I001


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class GalleryImage:
    title: str
    src: str
    alt: str
    article_url: str


class GalleryImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[GalleryImage] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "button":
            return
        data = {key: value or "" for key, value in attrs}
        src = data.get("data-zoom-src", "")
        if "/images/architectures/" not in src or "/thumbnails/" in src:
            return
        self.images.append(
            GalleryImage(
                title=data.get("data-zoom-title", ""),
                src=urljoin(GALLERY_URL, src),
                alt=data.get("data-zoom-alt", ""),
                article_url=data.get("data-zoom-article-url", ""),
            )
        )


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=60) as response:
        return response.read()


def gallery_images() -> list[GalleryImage]:
    parser = GalleryImageParser()
    parser.feed(fetch_bytes(GALLERY_URL).decode("utf-8"))
    return parser.images


def image_cache_path(url: str) -> Path:
    return CACHE_DIR / Path(url).name


def cached_image(url: str, *, refresh: bool) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = image_cache_path(url)
    if refresh or not path.exists():
        path.write_bytes(fetch_bytes(url))
    return path


def local_image(path: Path, filename: str) -> Path:
    source_path = path / filename
    if not source_path.exists():
        raise FileNotFoundError(f"missing local architecture image: {source_path}")
    return source_path


def image_dimensions(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["magick", str(path), "-format", "%w %h", "info:"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    width_text, height_text = result.stdout.split()
    return int(width_text), int(height_text)


def svg_text(*, title: str, source_url: str, image: bytes, width: int, height: int) -> str:
    encoded = base64.b64encode(image).decode("ascii")
    safe_title = html.escape(title, quote=False)
    safe_source = html.escape(source_url, quote=True)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img">\n'
        f"  <title>{safe_title}</title>\n"
        f'  <metadata source="{safe_source}" />\n'
        f'  <image href="data:image/webp;base64,{encoded}" '
        f'x="0" y="0" width="{width}" height="{height}" />\n'
        "</svg>\n"
    )


def build_asset(
    entry: registry.Entry,
    item: GalleryImage,
    *,
    refresh: bool,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    source_path = (
        local_image(source_dir, Path(item.src).name)
        if source_dir is not None
        else cached_image(item.src, refresh=refresh)
    )
    source_bytes = source_path.read_bytes()
    width, height = image_dimensions(source_path)

    svg_path = ASSET_DIR / f"{entry.slug}.svg"
    svg_path.write_text(
        svg_text(title=entry.name, source_url=item.src, image=source_bytes, width=width, height=height),
        encoding="utf-8",
    )
    return {
        "slug": entry.slug,
        "name": entry.name,
        "source_title": item.title,
        "source_alt": item.alt,
        "source_url": item.src,
        "article_url": item.article_url,
        "path": str(svg_path.relative_to(WEB_ROOT)),
        "width": width,
        "height": height,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def manifest_images() -> list[GalleryImage]:
    if not MANIFEST_PATH.exists():
        return []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    images: list[GalleryImage] = []
    for entry in registry.REGISTRY:
        asset = manifest.get("assets", {}).get(entry.slug)
        if not asset:
            return []
        images.append(
            GalleryImage(
                title=asset.get("source_title", entry.name),
                src=asset.get("source_url", ""),
                alt=asset.get("source_alt", ""),
                article_url=asset.get("article_url", ""),
            )
        )
    return images


def source_images(*, source_dir: Path | None) -> list[GalleryImage]:
    if source_dir is None:
        return gallery_images()
    images = manifest_images()
    if images:
        return images
    return gallery_images()


def build_manifest(*, refresh: bool, source_dir: Path | None = None) -> dict[str, Any]:
    images = source_images(source_dir=source_dir)
    entries = registry.REGISTRY
    if len(images) != len(entries):
        raise RuntimeError(f"gallery has {len(images)} architecture images, registry has {len(entries)}")

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    assets = [
        build_asset(entry, item, refresh=refresh, source_dir=source_dir)
        for entry, item in zip(entries, images, strict=True)
    ]
    return {
        "generated_by": "scripts/sync_architecture_assets.py",
        "source_gallery": GALLERY_URL,
        "asset_count": len(assets),
        "assets": {asset["slug"]: asset for asset in assets},
    }


def write_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_manifest() -> list[str]:
    if not MANIFEST_PATH.exists():
        return [f"missing {MANIFEST_PATH.relative_to(ROOT)}"]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assets = manifest.get("assets", {})
    expected_slugs = [entry.slug for entry in registry.REGISTRY]
    problems: list[str] = []
    if manifest.get("asset_count") != len(expected_slugs):
        problems.append("asset_count does not match registry")
    if list(assets) != expected_slugs:
        problems.append("manifest asset order does not match registry")
    for entry in registry.REGISTRY:
        asset = assets.get(entry.slug)
        if not asset:
            problems.append(f"missing asset for {entry.slug}")
            continue
        svg_path = WEB_ROOT / asset.get("path", "")
        if not svg_path.exists():
            problems.append(f"missing {svg_path.relative_to(ROOT)}")
            continue
        text = svg_path.read_text(encoding="utf-8", errors="replace")
        if "<svg" not in text or "data:image/webp;base64," not in text:
            problems.append(f"{svg_path.relative_to(ROOT)} is not a self-contained raster SVG")
            continue
        match = EMBEDDED_WEBP_RE.search(text)
        if not match:
            problems.append(f"{svg_path.relative_to(ROOT)} does not embed a WebP payload")
            continue
        try:
            embedded = base64.b64decode(match.group(1), validate=True)
        except ValueError:
            problems.append(f"{svg_path.relative_to(ROOT)} has invalid embedded WebP base64")
            continue
        if hashlib.sha256(embedded).hexdigest() != asset.get("source_sha256"):
            problems.append(f"{entry.slug} embedded WebP hash does not match manifest")
        if asset.get("width", 0) <= 0 or asset.get("height", 0) <= 0:
            problems.append(f"{entry.slug} has invalid dimensions")
        if not asset.get("source_url", "").endswith(".webp"):
            problems.append(f"{entry.slug} has invalid source_url")
        if len(asset.get("source_sha256", "")) != 64:
            problems.append(f"{entry.slug} has invalid source_sha256")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="fetch images and regenerate SVG assets")
    parser.add_argument("--check", action="store_true", help="validate committed assets without rewriting")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="regenerate SVG wrappers from local .webp files instead of fetching the gallery",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = validate_manifest()
        if problems:
            print("Architecture assets are not up to date:")
            for problem in problems:
                print(f"  - {problem}")
            print("Run: python scripts/sync_architecture_assets.py --refresh")
            return 1
        print("Architecture assets are up to date.")
        return 0

    manifest = build_manifest(refresh=args.refresh, source_dir=args.source_dir)
    write_manifest(manifest)
    print(f"Wrote {manifest['asset_count']} SVG assets to {ASSET_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
