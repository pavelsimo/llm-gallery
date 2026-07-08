# TODO

Improvements to the codebase, roughly in priority order.

## 1. Audit model implementations against architecture diagrams

Verify that each implementation in `llm_gallery/models/` is solid by cross-checking it
against its architecture diagram in `web/assets/architectures/<slug>.svg` (the diagrams
are SVGs with embedded raster images — there are no separate `.png` files) and against
the published specs in the model's tech report.

Things to check per model:

- [ ] Attention variant (MHA / GQA / MQA / MLA / linear / sliding window) matches the diagram
- [ ] Norm type and placement (pre/post, RMSNorm vs LayerNorm, QK-norm) matches
- [ ] Positional encoding (RoPE variant, NoPE layers, partial rotary) matches
- [ ] MoE details where applicable (expert count, shared experts, routing, top-k)
- [ ] FFN type and activation (SwiGLU / GeGLU / ReLU²) matches
- [ ] Config values (`n_layer`, `n_embd`, heads, vocab, context length) match the published spec
- [ ] Section docstrings / anchors used by the visualizer still point at the right code

Suggested order — tier 1 first (from `llm_gallery/models/registry.py`):

- [ ] gpt2-xl
- [ ] llama3-8b
- [ ] deepseek-v3
- [ ] deepseek-r1
- [ ] gemma3-27b
- [ ] qwen3-next-80b-a3b
- [ ] kimi-linear
- [ ] xlstm-7b
- [ ] nemotron3-nano-30b

Then tier 2, then tier 3 (tier 3 files are mostly config variants — verify configs only).

## 2. Tabbed Code / Tech Report view in model detail

Replace the external "Tech report" link with tabs in the model detail view so the user
can switch between reading the code and reading the tech report in place.

- [ ] Add a tab bar to `web/viewer.html` (e.g. **Code** | **Tech Report**) above the code pane
- [ ] Remove the plain external link rendering in `web/app.js:317` (`link("Tech report", ...)`)
- [ ] Tech Report tab embeds the report from the model JSON's `links.tech_report`
- [ ] Caveat: many report hosts (arxiv abstract pages, blogs) block iframes via
      `X-Frame-Options` / CSP. For arxiv, embed the PDF URL (`arxiv.org/pdf/...`) instead of
      the abstract page; for anything that can't be embedded, show a fallback panel with an
      "Open report ↗" link
- [ ] Keep the tab state per model (default to Code)

## 3. Remove search bar and badges from model detail view

The in-viewer code search isn't pulling its weight, and the badge pills under the model
header add noise.

- [ ] Remove the code search input (`web/viewer.html:52-56`, `#codeSearch`)
- [ ] Remove its wiring in `web/app.js`: input handler (~lines 446-449), `applyCodeSearch()`
      (~lines 525-558), and the `/` keyboard shortcut (~line 484)
- [ ] Remove the tier/template/config pills rendered in `web/app.js:298-307` into the
      `model-facts` section
- [ ] Clean up now-unused CSS (`.code-search`, `.pill`, `search-dim`) in `web/styles.css`

## 4. Remove learning path from main view

Which order to study the models in is up to the user — drop the guided rail.

- [ ] Remove the `#learningRail` section from `web/index.html:62`
- [ ] Remove `renderLearningRail()` and its call site in `web/app.js` (~lines 145-161)
- [ ] Remove the `.learning-rail` styles in `web/styles.css` (~lines 339-400)
- [ ] Optionally stop emitting `learning_path` / `path_order` in
      `scripts/build_visualizer_data.py` (~lines 2137-2142) and regenerate `web/data/`

## 5. Deploy the site to GitHub Pages via CI/CD

The site is fully static (`web/` + generated `web/data/`), so it can be published as-is.

- [ ] Add `.github/workflows/deploy.yml`:
  - Trigger on push to `main` (plus `workflow_dispatch`)
  - Set up Python + `uv`, install deps
  - Run `uv run python scripts/build_visualizer_data.py` to regenerate `web/data/`
  - Upload `web/` with `actions/upload-pages-artifact` and deploy with `actions/deploy-pages`
- [ ] Enable Pages in the repo settings (Source: GitHub Actions) at
      `github.com/pavelsimo/llm-gallery`
- [ ] Verify all asset/data paths in the HTML/JS are relative so the site works under
      `pavelsimo.github.io/llm-gallery/`
