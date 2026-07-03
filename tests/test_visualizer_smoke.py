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
            page.goto(f"{base_url}/viewer.html?model=llama3-8b", wait_until="networkidle")

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
