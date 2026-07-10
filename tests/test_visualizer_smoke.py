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


def test_theme_follows_system_persists_and_filters_artwork():
    with static_web_server() as base_url, sync_playwright() as playwright:
        browser = launch_chromium(playwright)
        try:
            context = browser.new_context(color_scheme="light")
            page = context.new_page()
            page.goto(f"{base_url}/index.html", wait_until="networkidle")
            expect(page.locator("html")).to_have_attribute("data-theme", "light")
            toggle = page.locator(".theme-toggle")
            expect(toggle).to_have_attribute("aria-pressed", "false")
            expect(toggle).to_have_attribute("aria-label", "Switch to dark mode")

            page.emulate_media(color_scheme="dark")
            expect(page.locator("html")).to_have_attribute("data-theme", "dark")
            page.emulate_media(color_scheme="light")
            expect(page.locator("html")).to_have_attribute("data-theme", "light")

            toggle.click()
            expect(page.locator("html")).to_have_attribute("data-theme", "dark")
            expect(toggle).to_have_attribute("aria-pressed", "true")
            assert page.evaluate("() => localStorage.getItem('llm-gallery-theme')") == "dark"
            page.emulate_media(color_scheme="light")
            expect(page.locator("html")).to_have_attribute("data-theme", "dark")

            page.goto(f"{base_url}/viewer.html?model=kimi-k2", wait_until="networkidle")
            expect(page.locator("html")).to_have_attribute("data-theme", "dark")
            expect(page.locator(".theme-toggle")).to_have_count(0)
            artwork = page.locator(".diagram-artwork-image")
            expect(artwork).to_be_visible()
            assert "invert(1)" in artwork.evaluate("node => getComputedStyle(node).filter")

            page.goto(f"{base_url}/index.html", wait_until="networkidle")
            page.locator(".theme-toggle").click()
            expect(page.locator("html")).to_have_attribute("data-theme", "light")
            page.goto(f"{base_url}/viewer.html?model=kimi-k2", wait_until="networkidle")
            artwork = page.locator(".diagram-artwork-image")
            page.wait_for_timeout(250)
            assert artwork.evaluate("node => getComputedStyle(node).filter") == "none"

            page.goto(f"{base_url}/compare.html?left=kimi-k2&right=llama3-8b", wait_until="networkidle")
            expect(page.locator("html")).to_have_attribute("data-theme", "light")
            expect(page.locator(".theme-toggle")).to_have_count(0)
            expect(page.locator(".diagram-artwork-image")).to_have_count(2)
            assert page.locator(".diagram-artwork-image").first.evaluate(
                "node => getComputedStyle(node).filter"
            ) == "none"
            context.close()

            dark_context = browser.new_context(color_scheme="dark")
            dark_page = dark_context.new_page()
            dark_page.goto(f"{base_url}/index.html", wait_until="networkidle")
            expect(dark_page.locator("html")).to_have_attribute("data-theme", "dark")
            dark_context.close()

            fallback_context = browser.new_context(color_scheme="light")
            fallback_context.add_init_script(
                """
                Object.defineProperty(window, "localStorage", {
                  get() { throw new DOMException("Storage unavailable", "SecurityError"); }
                });
                """
            )
            fallback_page = fallback_context.new_page()
            fallback_page.goto(f"{base_url}/index.html", wait_until="networkidle")
            expect(fallback_page.locator("html")).to_have_attribute("data-theme", "light")
            fallback_page.locator(".theme-toggle").click()
            expect(fallback_page.locator("html")).to_have_attribute("data-theme", "dark")
            fallback_context.close()

            mobile_context = browser.new_context(
                color_scheme="light", viewport={"width": 390, "height": 844}
            )
            mobile_page = mobile_context.new_page()
            mobile_page.goto(f"{base_url}/viewer.html?model=kimi-k2", wait_until="networkidle")
            expect(mobile_page.locator(".theme-toggle")).to_have_count(0)
            assert mobile_page.evaluate(
                "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth"
            )
            mobile_context.close()
        finally:
            browser.close()


def test_viewer_diagram_code_roundtrip():
    with static_web_server() as base_url, sync_playwright() as playwright:
        browser = launch_chromium(playwright)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            page.goto(f"{base_url}/index.html", wait_until="networkidle")
            expect(page.locator(".hero h1")).to_have_text("LLM gallery")
            expect(page.locator(".hero h1 em")).to_have_text("gallery")
            expect(page.locator(".hero-attribution")).to_contain_text("Based on Sebastian Raschka")
            assert page.locator("text=Every architecture in the gallery").count() == 0
            github_icon = page.locator(".header-link-icon")
            expect(github_icon).to_be_visible()
            github_icon_box = github_icon.bounding_box()
            assert github_icon_box is not None
            assert github_icon_box["width"] <= 24
            assert github_icon_box["height"] <= 24

            page.goto(f"{base_url}/viewer.html?model=llama3-8b", wait_until="networkidle")
            assert page.locator(".terminal-bar .dot").count() == 0
            assert page.locator("#copyActiveLink").count() == 0
            assert page.locator("#copySection").count() == 0
            assert page.locator("#copyCode").count() == 0
            expect(page.locator("#codeTab")).to_have_attribute("aria-selected", "true")
            expect(page.locator("#codeTabPanel")).to_be_visible()
            expect(page.locator("#reportTabPanel")).to_be_hidden()
            assert page.locator("#galleryTab").count() == 0
            assert page.locator("#galleryTabPanel").count() == 0
            expect(page.locator(".diagram-artwork-image")).to_be_visible()
            expect(page.locator("#modelFacts")).to_contain_text("GQA + RoPE + SwiGLU + RMSNorm")
            expect(page.locator("#modelFacts")).to_contain_text("Released 2024-04-18")
            gallery_source = page.locator("#modelFacts .fact-links a")
            expect(gallery_source).to_be_visible()
            expect(gallery_source).to_have_text("More information")
            expect(gallery_source).to_have_attribute("aria-label", "More information")
            expect(gallery_source).to_have_attribute("title", "More information")
            gallery_source_href = gallery_source.get_attribute("href")
            assert gallery_source_href is not None
            assert gallery_source_href.startswith("https://")
            assert page.evaluate(
                """
                () => {
                  const diagram = document.querySelector("#diagramWrap").getBoundingClientRect();
                  const facts = document.querySelector("#modelFacts").getBoundingClientRect();
                  const related = document.querySelector("#relatedModels").getBoundingClientRect();
                  return facts.top >= diagram.bottom && related.top >= facts.bottom;
                }
                """
            )

            page.locator("#reportTab").click()
            expect(page.locator("#reportTabPanel")).to_be_visible()

            page.locator("#codeTab").click()
            expect(page.locator("#codeTabPanel")).to_be_visible()
            expect(page.locator("#reportTabPanel")).to_be_hidden()
            page.locator("#codeTab").focus()
            page.keyboard.press("ArrowRight")
            expect(page.locator("#learnTabPanel")).to_be_visible()
            page.keyboard.press("ArrowRight")
            expect(page.locator("#reportTabPanel")).to_be_visible()
            page.keyboard.press("ArrowRight")
            expect(page.locator("#codeTabPanel")).to_be_visible()
            page.keyboard.press("ArrowLeft")
            expect(page.locator("#reportTabPanel")).to_be_visible()
            page.locator("#codeTab").click()

            page.goto(f"{base_url}/viewer.html?model=llama3-8b&tab=gallery", wait_until="networkidle")
            expect(page.locator("#codeTab")).to_have_attribute("aria-selected", "true")
            expect(page.locator("#codeTabPanel")).to_be_visible()
            assert "tab=gallery" not in page.url

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
                  const usage = target.ranges.filter((range) => range.kind === "usage");
                  return { lineStart: target.line_start, lineEnd: target.line_end, usage };
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

            assert target["usage"], "model.lm-head should carry at least one usage range"
            assert page.evaluate(
                """
                (usage) => {
                  const highlighted = [...document.querySelectorAll(".code-line.usage-active")]
                    .map((line) => Number(line.dataset.line));
                  const expected = usage.flatMap((range) => {
                    const lines = [];
                    for (let line = range.line_start; line <= range.line_end; line += 1) lines.push(line);
                    return lines;
                  });
                  return JSON.stringify(highlighted) === JSON.stringify(expected);
                }
                """,
                target["usage"],
            )

            usage_line = target["usage"][0]["line_start"]
            page.locator(f'.code-line[data-line="{usage_line}"]').click()
            assert page.evaluate("() => window.currentTargetId") == "model.lm-head"

            attention_line = page.evaluate(
                """
                () => window.currentModel.sections.find((section) => section.role === "attention").line_start
                """
            )
            page.locator(f'.code-line[data-line="{attention_line}"]').click()
            assert page.evaluate("() => window.currentTargetId") == "attention"
            expect(page.locator('.diagram-hotspot.active[data-target-id="attention"]').first).to_be_visible()

            # Learn tab: concept explanation follows the active target.
            page.locator("#learnTab").click()
            expect(page.locator("#learnTabPanel")).to_be_visible()
            expect(page.locator("#codeTabPanel")).to_be_hidden()
            assert "tab=learn" in page.url
            expect(page.locator("#conceptTitle")).not_to_have_text("")
            attention_concept_title = page.locator("#conceptTitle").inner_text()
            assert page.locator("#conceptBody .katex").count() > 0, "KaTeX did not render any equations"
            assert page.locator("#conceptIndex .concept-chip").count() > 0

            page.locator('.diagram-hotspot[data-target-id="model.lm-head"]').click()
            assert page.evaluate("() => window.currentTargetId") == "model.lm-head"
            expect(page.locator("#conceptTitle")).not_to_have_text(attention_concept_title)

            # Concept-index chips round-trip into diagram/code selection.
            page.locator("#conceptIndex .concept-chip").first.click()
            assert page.evaluate("() => window.currentTargetId") == "config"
            assert page.evaluate(
                """
                () => {
                  const config = window.currentModel.sections.find((section) => section.id === "config");
                  const active = [...document.querySelectorAll(".code-line.active")]
                    .map((line) => Number(line.dataset.line));
                  return active[0] === config.line_start && active.at(-1) === config.line_end;
                }
                """
            )

            page.locator("#codeTab").click()
            assert "tab=learn" not in page.url

            # Deep link straight into the Learn tab.
            page.goto(f"{base_url}/viewer.html?model=llama3-8b&tab=learn&section=attention", wait_until="networkidle")
            expect(page.locator("#learnTabPanel")).to_be_visible()
            expect(page.locator("#conceptTitle")).not_to_have_text("")
            assert page.locator("#conceptBody .katex").count() > 0
        finally:
            browser.close()
