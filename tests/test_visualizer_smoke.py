from __future__ import annotations

import contextlib
import functools
import http.server
import socket
import threading
from pathlib import Path

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"


@contextlib.contextmanager
def static_web_server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB_DIR))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        host, port = probe.getsockname()

    server = http.server.ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def launch_chromium(playwright):
    try:
        return playwright.chromium.launch()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "Please run the following command" in str(exc):
            pytest.skip("Playwright Chromium is not installed; run `uv run playwright install chromium`")
        raise


def test_viewer_diagram_code_roundtrip():
    with static_web_server() as base_url, sync_playwright() as playwright:
        browser = launch_chromium(playwright)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            page.goto(f"{base_url}/index.html", wait_until="networkidle")
            assert page.locator("text=Every architecture in the gallery").count() == 0
            github_icon = page.locator(".header-link-icon")
            expect(github_icon).to_be_visible()
            github_icon_box = github_icon.bounding_box()
            assert github_icon_box is not None
            assert github_icon_box["width"] <= 24
            assert github_icon_box["height"] <= 24

            page.evaluate("""() => localStorage.setItem("llm-gallery-viewer-tab:llama3-8b", "report")""")
            page.goto(f"{base_url}/viewer.html?model=llama3-8b", wait_until="networkidle")
            assert page.locator(".terminal-bar .dot").count() == 0
            assert page.locator("#copyActiveLink").count() == 0
            assert page.locator("#copySection").count() == 0
            assert page.locator("#copyCode").count() == 0
            expect(page.locator("#codeTab")).to_have_attribute("aria-selected", "true")
            expect(page.locator("#codeTabPanel")).to_be_visible()
            expect(page.locator("#reportTabPanel")).to_be_hidden()
            gallery_tab = page.locator("#galleryTab")
            expect(gallery_tab).to_be_visible()
            assert gallery_tab.get_attribute("href") is None
            assert gallery_tab.get_attribute("target") is None
            expect(page.locator("#galleryTabPanel")).to_be_hidden()

            popups = []
            page.on("popup", lambda popup: popups.append(popup))
            viewer_url = page.url
            gallery_tab.click()
            page.wait_for_timeout(100)
            assert popups == []
            assert page.url == viewer_url
            expect(page.locator("#galleryTab")).to_have_attribute("aria-selected", "true")
            expect(page.locator("#galleryTabPanel")).to_be_visible()
            expect(page.locator("#codeTabPanel")).to_be_hidden()
            expect(page.locator("#reportTabPanel")).to_be_hidden()
            expect(page.locator("#galleryArtwork")).to_be_visible()
            assert "assets/architectures/llama3-8b.svg" in (page.locator("#galleryArtwork").get_attribute("src") or "")
            gallery_source = page.locator("#gallerySourceLink")
            expect(gallery_source).to_be_visible()
            gallery_source_href = gallery_source.get_attribute("href")
            assert gallery_source_href is not None
            assert gallery_source_href.startswith("https://")

            page.locator("#reportTab").click()
            expect(page.locator("#reportTabPanel")).to_be_visible()
            expect(page.locator("#galleryTabPanel")).to_be_hidden()

            page.locator("#codeTab").click()
            expect(page.locator("#codeTabPanel")).to_be_visible()
            expect(page.locator("#reportTabPanel")).to_be_hidden()
            expect(page.locator("#galleryTabPanel")).to_be_hidden()
            page.locator("#codeTab").focus()
            page.keyboard.press("ArrowRight")
            expect(page.locator("#reportTabPanel")).to_be_visible()
            page.keyboard.press("ArrowRight")
            expect(page.locator("#galleryTabPanel")).to_be_visible()
            page.keyboard.press("ArrowLeft")
            expect(page.locator("#reportTabPanel")).to_be_visible()
            page.locator("#codeTab").click()

            expect(page.locator(".diagram-artwork-image")).to_be_visible()
            assert page.locator(".diagram-hotspot").count() > 0
            assert page.locator(".diagram-node").count() == 0
            diagram_target = page.locator('.diagram-hotspot[data-target-id="model.lm-head"]')
            expect(diagram_target).to_be_visible()
            diagram_target.click()

            target = page.evaluate(
                """
                () => {
                  const target = window.currentModel.anchors.find((anchor) => anchor.id === "model.lm-head");
                  return { lineStart: target.line_start, lineEnd: target.line_end };
                }
                """
            )
            assert page.evaluate("() => window.currentTargetId") == "model.lm-head"
            assert "section=model&target=model.lm-head" in page.url
            assert page.evaluate(
                """
                ({ lineStart, lineEnd }) => {
                  const active = [...document.querySelectorAll(".code-line.active")]
                    .map((line) => Number(line.dataset.line));
                  return (
                    active.length === lineEnd - lineStart + 1 &&
                    active[0] === lineStart &&
                    active.at(-1) === lineEnd
                  );
                }
                """,
                target,
            )
            assert page.evaluate(
                """
                (lineStart) => {
                  const line = document.querySelector(`.code-line[data-line="${lineStart}"]`);
                  const codeBlock = document.querySelector(".code-block");
                  const lineRect = line.getBoundingClientRect();
                  const codeRect = codeBlock.getBoundingClientRect();
                  return lineRect.top >= codeRect.top && lineRect.bottom <= codeRect.bottom;
                }
                """,
                target["lineStart"],
            )
            assert page.evaluate(
                """
                (lineStart) => {
                  const line = document.querySelector(`.code-line[data-line="${lineStart}"]`);
                  const lineRect = line.getBoundingClientRect();
                  return lineRect.top >= 0 && lineRect.bottom <= window.innerHeight;
                }
                """,
                target["lineStart"],
            )

            attention_line = page.evaluate(
                """
                () => window.currentModel.sections.find((section) => section.role === "attention").line_start
                """
            )
            page.locator(f'.code-line[data-line="{attention_line}"]').click()
            assert page.evaluate("() => window.currentTargetId") == "attention"
            expect(page.locator('.diagram-hotspot.active[data-target-id="attention"]').first).to_be_visible()
        finally:
            browser.close()
