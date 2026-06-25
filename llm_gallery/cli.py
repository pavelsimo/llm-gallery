"""Command-line entry point: list | info | smoke | train | generate.

    uv run python -m llm_gallery.cli list
    uv run python -m llm_gallery.cli info gpt2-xl
    uv run python -m llm_gallery.cli smoke gpt2-xl
    uv run python -m llm_gallery.cli train gpt2-xl --steps 500
    uv run python -m llm_gallery.cli generate gpt2-xl --ckpt checkpoints/gpt2-xl.pt --prompt "ROMEO:"

The CLI is the one place that knows about both the model contract and the harness; model files stay
unaware of either.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .harness import data as data_mod
from .harness.generate import generate
from .harness.interface import count_params, human
from .harness.train import TrainConfig, train
from .models import registry

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _build(slug: str, preset: str, **overrides):
    """Import a model module and build (module, config, model) for the given preset."""
    module = registry.load(slug)
    if preset not in module.PRESETS:
        raise SystemExit(
            f"unknown preset {preset!r} for {slug}; available: {list(module.PRESETS)}"
        )
    cfg = module.PRESETS[preset]
    if overrides:
        cfg = replace(cfg, **overrides)
    model = module.Model(cfg)
    return module, cfg, model


# --------------------------------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------------------------------
# Pedagogical notes shown by `path` — one line per model, explaining what's new vs the prior tier.
_PATH_NOTES: dict[str, str] = {
    # Tier 1
    "gpt2-xl":             "learned-abs positions · MHA · GELU MLP · LayerNorm — the classic baseline",
    "llama3-8b":           "RoPE · GQA · SwiGLU · RMSNorm (pre-norm) — the modern dense template",
    "gemma3-27b":          "sliding-window attn · sandwich norm · QK-Norm · GeGLU · decoupled head_dim",
    "deepseek-v3":         "Multi-head Latent Attention (MLA) · fine-grained MoE · shared always-on expert",
    "xlstm-7b":            "mLSTM matrix memory — NO attention; fully recurrent over time",
    "nemotron3-nano-30b":  "Mamba-2 SSM layers interleaved with full-attention layers",
    "qwen3-next-80b-a3b":  "Gated DeltaNet (linear attn) interleaved with gated full-attention",
    "kimi-linear":         "Lightning linear attention · MLA · hybrid linear/full-attention",
    # Tier 2
    "llama3.2-1b":         "1B Llama — good scale comparison point with llama3-8b",
    "olmo2-7b":            "post-norm placement · QK-Norm — contrast pre-norm vs post-norm",
    "deepseek-r1":         "same MLA + MoE as V3, trained for chain-of-thought reasoning",
    "mistral-small-3.1":   "GQA dense — clean secondary reference for the Llama recipe",
    "qwen3-0.6b":          "Llama recipe + per-head QK-Norm · 0.6B scale for fast iteration",
    "qwen3-30b-a3b":       "top-k sparse routing · load-balance loss — standard MoE baseline",
    "smollm3-3b":          "periodic NoPE: some layers omit positional encoding entirely",
    "gpt-oss-20b":         "attention sinks · alternating global/local attention layer pattern",
    "tiny-aya":            "parallel blocks: attention ∥ FFN share the same residual (not sequential)",
    "phi-4":               "compact 14B dense GQA from Microsoft — clean modern academic baseline",
}


def cmd_path(args: argparse.Namespace) -> None:
    header = "Learning Path — llm-gallery"
    print(f"\n{header}")
    print("=" * len(header))

    tier_labels = {
        1: "Tier 1 · Essential — start here, read in order",
        2: "Tier 2 · Important — once you've read Tier 1",
        3: "Tier 3 · Variants — read the base file instead",
    }
    tier_descs = {
        1: "Each file introduces a distinct architecture family.",
        2: "Adds secondary innovations not covered by Tier 1.",
        3: f"Size/config variants of Tier 1/2 models (use `llm-gallery list` to browse).",
    }

    max_tiers = 3 if args.all else 2
    for t in range(1, max_tiers + 1):
        entries = registry.tier_entries(t)
        if not entries:
            continue
        label = tier_labels[t]
        desc = tier_descs[t]
        print(f"\n{label}  ({len(entries)} models)")
        print(f"  {desc}")
        print()
        if t == 3 and not args.all:
            continue
        slug_w = max(len(e.slug) for e in entries) + 2
        name_w = max(len(e.name) for e in entries) + 2
        for i, e in enumerate(entries, 1):
            note = _PATH_NOTES.get(e.slug, e.archetype)
            if t == 1:
                print(f"  {i:2}.  {e.slug:<{slug_w}}  {e.name:<{name_w}}  {note}")
            else:
                print(f"       {e.slug:<{slug_w}}  {e.name:<{name_w}}  {note}")

    if not args.all:
        n3 = len(registry.tier_entries(3))
        print(f"\n       ({n3} variant files — pass --all to list them)")
    print()
    print("  Tips:")
    print("    llm-gallery info <slug>   — show module docstring + tech-report URL")
    print("    llm-gallery smoke <slug>  — run a quick forward/backward sanity check")
    print("    llm-gallery list          — see every model with status and archetype")
    print()


def cmd_list(args: argparse.Namespace) -> None:
    rows = registry.REGISTRY
    if args.done:
        rows = [e for e in rows if e.status == registry.DONE]
    width = max(len(e.slug) for e in rows)
    done = sum(e.status == registry.DONE for e in registry.REGISTRY)
    print(f"{len(registry.REGISTRY)} models in the gallery — {done} implemented\n")
    for e in rows:
        mark = "✓" if e.status == registry.DONE else " "
        print(f" [{mark}] {e.slug:<{width}}  {e.release:<10}  {e.archetype}")


def cmd_info(args: argparse.Namespace) -> None:
    entry = registry.get(args.slug)
    print(f"{entry.name}\n  slug     : {entry.slug}\n  release  : {entry.release or 'n/a'}")
    print(f"  archetype: {entry.archetype}\n  status   : {entry.status}")
    if entry.status != registry.DONE:
        print("\n(not implemented yet)")
        return
    module = registry.load(args.slug)
    for field in ("TECH_REPORT_URL", "GALLERY_URL"):
        if hasattr(module, field):
            print(f"  {field.lower():9}: {getattr(module, field)}")
    print(f"  presets  : {list(module.PRESETS)}")
    if module.__doc__:
        print("\n" + module.__doc__.strip())


def cmd_smoke(args: argparse.Namespace) -> None:
    module, cfg, model = _build(args.slug, args.preset)
    model.to(args.device)
    T = min(16, cfg.context_length)
    idx = torch.randint(0, cfg.vocab_size, (2, T), device=args.device)
    logits = model(idx)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), idx.reshape(-1)
    )
    loss.backward()
    grads_ok = all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    print(f"{module.MODEL_NAME}  (preset={args.preset})")
    print(f"  params  : {count_params(model):,}  ({human(count_params(model))})")
    print(f"  input   : {tuple(idx.shape)}")
    print(f"  logits  : {tuple(logits.shape)}  (expected (2, {T}, {cfg.vocab_size}))")
    print(f"  loss    : {loss.item():.4f}")
    print(f"  grads   : {'all finite' if grads_ok else 'NON-FINITE!'}")


def cmd_train(args: argparse.Namespace) -> None:
    dataset = data_mod.load_char_shakespeare()
    # The model must use the dataset's vocabulary; everything else comes from the preset.
    _, cfg, model = _build(args.slug, args.preset, vocab_size=dataset.vocab_size)
    block_size = min(args.block_size, cfg.context_length)
    if block_size != args.block_size:
        print(f"[train] clamping block_size {args.block_size} -> {block_size} (context limit)")

    print(f"training {args.slug} (preset={args.preset}) | params={human(count_params(model))} "
          f"| vocab={dataset.vocab_size} | device={args.device}")
    tcfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=block_size,
        device=args.device,
    )
    ckpt = Path(args.ckpt) if args.ckpt else CKPT_DIR / f"{args.slug}.pt"
    train(model, dataset, tcfg, ckpt_path=ckpt, extra_state={"config": asdict(cfg), "slug": args.slug})


def cmd_generate(args: argparse.Namespace) -> None:
    if args.ckpt:
        # Our checkpoints only contain tensors + plain dicts (state_dict, stoi/itos, config),
        # so the safe loader works and avoids unpickling arbitrary objects.
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
        module = registry.load(args.slug)
        cfg = module.Config(**ckpt["config"])
        model = module.Model(cfg).to(args.device)
        model.load_state_dict(ckpt["model"])
        stoi, itos = ckpt["stoi"], ckpt["itos"]
    else:
        # No checkpoint: build a fresh (untrained) model on the shakespeare vocab. Output will be
        # gibberish, but it exercises the full sampling path.
        print("[generate] no --ckpt given; sampling from an UNTRAINED model (expect gibberish)")
        dataset = data_mod.load_char_shakespeare()
        _, cfg, model = _build(args.slug, args.preset, vocab_size=dataset.vocab_size)
        model.to(args.device)
        stoi, itos = dataset.stoi, dataset.itos

    encode = [stoi.get(c, 0) for c in args.prompt]
    idx = torch.tensor([encode], dtype=torch.long, device=args.device)
    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        context_length=cfg.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    text = "".join(itos[int(i)] for i in out[0].tolist())
    print("-" * 60)
    print(text)
    print("-" * 60)


# --------------------------------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm_gallery", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("path", help="show the recommended learning path through the gallery")
    pp.add_argument("--all", action="store_true", help="include Tier 3 variant files in the output")
    pp.set_defaults(func=cmd_path)

    pl = sub.add_parser("list", help="list all models in the gallery")
    pl.add_argument("--done", action="store_true", help="show only implemented models")
    pl.set_defaults(func=cmd_list)

    pi = sub.add_parser("info", help="show a model's architecture notes and links")
    pi.add_argument("slug")
    pi.set_defaults(func=cmd_info)

    ps = sub.add_parser("smoke", help="forward+backward a tiny build and report shapes")
    ps.add_argument("slug")
    ps.add_argument("--preset", default="tiny")
    ps.add_argument("--device", default="cpu")
    ps.set_defaults(func=cmd_smoke)

    pt = sub.add_parser("train", help="train a model on char-level tiny-shakespeare")
    pt.add_argument("slug")
    pt.add_argument("--preset", default="tiny")
    pt.add_argument("--steps", type=int, default=1000)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--block-size", type=int, default=128)
    pt.add_argument("--device", default="cpu")
    pt.add_argument("--ckpt", default=None, help="checkpoint path (default: checkpoints/<slug>.pt)")
    pt.set_defaults(func=cmd_train)

    pg = sub.add_parser("generate", help="sample text from a model")
    pg.add_argument("slug")
    pg.add_argument("--ckpt", default=None, help="trained checkpoint to load")
    pg.add_argument("--preset", default="tiny", help="preset to use when no --ckpt is given")
    pg.add_argument("--prompt", default="\n")
    pg.add_argument("--max-new-tokens", type=int, default=300)
    pg.add_argument("--temperature", type=float, default=0.8)
    pg.add_argument("--top-k", type=int, default=None)
    pg.add_argument("--top-p", type=float, default=None)
    pg.add_argument("--device", default="cpu")
    pg.set_defaults(func=cmd_generate)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
