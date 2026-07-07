# llm-gallery

PyTorch reimplementations of **every model** in Sebastian Raschka's
[LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery), built for **learning**
PyTorch and modern LLM architecture by reading and running clean code.

## Philosophy

- **One self-contained file per model.** Open any file in `llm_gallery/models/` and read the whole
  architecture top-to-bottom — every norm, attention, and feed-forward block is defined right there
  (nanoGPT style). Shared building blocks are intentionally duplicated across files so nothing is hidden
  behind imports.
- **Architecture, not weights.** Models are randomly initialized. There is no HuggingFace dependency and no
  pretrained-weight loading — the focus is the *mechanics*: shapes, forward/backward, parameter counts.
- **Runnable.** A small shared harness can train any model on a tiny dataset and sample text from it, so you
  can watch an architecture actually learn.

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

The repo also includes a static side-by-side model/code visualizer in `web/`:

```bash
uv run python scripts/build_visualizer_data.py
python -m http.server 8000 --directory web
```

Open `http://localhost:8000` to browse every registered model. The generated JSON in `web/data/`
maps diagram blocks to exact source line ranges, so clicking a block highlights the corresponding
Python section and clicking code highlights the matching diagram node.

## Hardware notes

Developed against a GTX 1060 6GB (Pascal). There's **no hardware bf16** on Pascal, so the code defaults to
**fp32** and the runnable `tiny` presets are sized to train on CPU or a small GPU in minutes. The `real`
presets in each file encode the *published* dimensions for reference; they are documentation, not meant to be
instantiated full-size on modest hardware. For `real.context_length`, the convention is: use the public
HuggingFace `config.json` `max_position_embeddings` when available; otherwise use the gallery card's context
length. This may differ from the short `tiny` preset context and from rounded labels like "128K".

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
implemented archetype with an explicit `# ASSUMPTION:` / `NOTE:` in the file docstring. The `real`
presets encode published dimensions for reference; the runnable preset everywhere is `tiny`.
