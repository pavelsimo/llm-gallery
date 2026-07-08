# llm-gallery

PyTorch reimplementations of **every model** in Sebastian Raschka's
[LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery). The goal is to learn how
modern LLMs actually work by building each architecture from scratch, then reading, running, and training it.

## Philosophy

Every model lives in a single self-contained file. Open anything in `llm_gallery/models/` and you can read
the whole architecture top to bottom: every norm, attention, and feed-forward block is defined right there,
nanoGPT style. Building blocks are deliberately duplicated across files so nothing hides behind an import.

The focus is architecture, not weights. Models are randomly initialized, there is no HuggingFace dependency,
and no pretrained weights get loaded. What matters is the mechanics: shapes, the forward and backward pass,
parameter counts.

And everything runs. A small shared harness can train any model on a tiny dataset and sample text from it,
so you can watch an architecture actually learn.

## Quickstart

```bash
uv sync                                   # install torch, numpy, pytest, ruff

# show the recommended learning path (Tier 1 → 2 → 3)
uv run python -m llm_gallery.cli path

# list every model in the gallery and its status
uv run python -m llm_gallery.cli list

# read a model's architecture notes + paper link
uv run python -m llm_gallery.cli info gpt2-xl

# build the tiny preset, run forward+backward, print shapes & param count
uv run python -m llm_gallery.cli smoke gpt2-xl

# train the tiny preset on char-level tiny-shakespeare
uv run python -m llm_gallery.cli train gpt2-xl --preset tiny --steps 500

# sample text from the trained checkpoint
uv run python -m llm_gallery.cli generate gpt2-xl --ckpt checkpoints/gpt2-xl.pt --prompt "ROMEO:"

# run the lightweight parametrized test suite over every registered model
uv run pytest

# run real-scale meta-device parameter-count audits
uv run pytest -m slow
```

Every model file is also runnable on its own as a smoke test:

```bash
uv run python llm_gallery/models/gpt2_xl.py
```

## Interactive visualizer

The visualizer is live at **<https://pavelsimo.github.io/llm-gallery/>**. It shows each model's diagram and
source side by side: clicking a diagram block highlights the matching Python, and clicking code highlights
the matching diagram node.

To build and serve it locally:

```bash
uv run python scripts/build_visualizer_data.py
python -m http.server 8000 --directory web
```

## Presets

Each model file defines two presets. `tiny` is the runnable one, sized to train on a CPU or small GPU in
minutes; it's what the CLI, tests, and training harness use. `real` records the published dimensions for
reference and isn't meant to be instantiated full-size. For `real.context_length`, the convention is the
public HuggingFace `config.json` `max_position_embeddings` when available, otherwise the gallery card's
context length, which can differ from rounded labels like "128K".

## Layout

```
llm_gallery/
  models/        one self-contained file per gallery model + registry.py
  harness/       shared tooling: data.py, train.py, generate.py, interface.py
  cli.py         list | info | path | smoke | train | generate
docs/
  architectures.md   the taxonomy of ideas the 83 models are built from
tests/
  test_models.py     parametrized over the registry (shapes, backward, determinism, generate)
```

## Learning path

```bash
uv run python -m llm_gallery.cli path        # print the recommended reading order
uv run python -m llm_gallery.cli path --all  # include all 83 models
```

The 83 models are grouped into three tiers:

| Tier | Count | Meaning |
|------|-------|---------|
| **1 — Essential** | 8 | One file per architectural family; start here |
| **2 — Important** | 10 | Introduce a specific new idea vs. a Tier 1 neighbour |
| **3 — Variant** | 65 | Config/scale variants; read the base file first |

**Suggested workflow per model:**

1. `uv run python -m llm_gallery.cli info <slug>` — read the one-line concept note and paper link
2. Open `llm_gallery/models/<slug>.py` — every building block is in a single file, top-to-bottom
3. `uv run python -m llm_gallery.cli smoke <slug>` — run forward+backward; verify shapes and param count

See [`docs/architectures.md`](docs/architectures.md) for the conceptual "family tree" of building blocks
(normalization, positional encodings, attention variants, MoE, non-attention mixers).

## Status

**All 83 models are implemented** and covered by the test suite (`uv run pytest`). Run
`uv run python -m llm_gallery.cli list` for the per-model list.

How the models were built:
- The architecturally-distinct models are hand-written and heavily annotated:
  `gpt2_xl`, `llama3_8b`, `olmo2_7b`, `qwen3_0_6b`, `gemma3_27b`, `smollm3_3b` (dense);
  `qwen3_30b_a3b`, `gpt_oss_20b` (MoE); `deepseek_v3` (MLA); `xlstm_7b`, `nemotron3_nano_30b`,
  `qwen3_next_80b_a3b`, `kimi_linear` (non-attention mixers); `tiny_aya` (parallel blocks).
- The remaining models are **size/config variants** of those, generated by `scripts/_gen_dense.py`
  into fully self-contained files. Re-run it with `uv run python scripts/_gen_dense.py`.

A handful of long-tail entries have little public detail, or use mechanisms not yet implemented here
(e.g. sparse/compressed attention, latent MoE, short-conv blocks). Those are modeled as their nearest
implemented archetype with an explicit `# ASSUMPTION:` / `NOTE:` in the file docstring.
