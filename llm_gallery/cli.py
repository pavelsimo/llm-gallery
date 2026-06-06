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
