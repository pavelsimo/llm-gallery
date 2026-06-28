"""One test module, parametrized over every model marked ``done`` in the registry.

As you finish a model and flip its registry status to ``done``, these checks start covering it
automatically — a cheap but effective quality gate across all 83 implementations:

  * builds the ``tiny`` preset,
  * forward pass returns logits of shape [B, T, vocab] with finite values,
  * backward produces finite gradients,
  * the model is deterministic (same seed + input -> same output in eval mode),
  * the generation path runs end-to-end.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from llm_gallery.harness.generate import generate
from llm_gallery.harness.interface import count_params
from llm_gallery.models import registry

SLUGS = registry.done_slugs()


@pytest.fixture(params=SLUGS)
def built(request):
    """Build the tiny preset for one model. Function-scoped: each test gets a fresh model."""
    slug = request.param
    module = registry.load(slug)
    cfg = module.PRESETS["tiny"]
    torch.manual_seed(0)
    model = module.Model(cfg)
    return slug, module, cfg, model


def test_has_tiny_preset(built):
    _, module, cfg, _ = built
    assert "tiny" in module.PRESETS
    assert cfg.vocab_size > 0
    assert cfg.context_length > 0


def test_forward_shape(built):
    _, _, cfg, model = built
    b, t = 2, min(16, cfg.context_length)
    idx = torch.randint(0, cfg.vocab_size, (b, t))
    logits = model(idx)
    assert logits.shape == (b, t, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_backward_finite(built):
    _, _, cfg, model = built
    b, t = 2, min(16, cfg.context_length)
    idx = torch.randint(0, cfg.vocab_size, (b, t))
    logits = model(idx)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), idx.reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_param_count_small(built):
    _, _, _, model = built
    n = count_params(model)
    assert 0 < n < 50_000_000, f"tiny preset should be small, got {n:,} params"


def test_deterministic(built):
    _, module, cfg, _ = built
    b, t = 2, min(16, cfg.context_length)
    idx = torch.randint(0, cfg.vocab_size, (b, t))
    torch.manual_seed(0)
    m1 = module.Model(cfg).eval()
    torch.manual_seed(0)
    m2 = module.Model(cfg).eval()
    with torch.no_grad():
        assert torch.allclose(m1(idx), m2(idx), atol=1e-5)


def test_generate_runs(built):
    _, _, cfg, model = built
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = generate(model, idx, max_new_tokens=5, context_length=cfg.context_length, top_k=10)
    assert out.shape == (1, 6)
    assert int(out.max()) < cfg.vocab_size


def test_registry_is_consistent():
    """Slugs and module stems must be unique across the whole gallery."""
    slugs = [e.slug for e in registry.REGISTRY]
    modules = [e.module for e in registry.REGISTRY]
    assert len(slugs) == len(set(slugs)), "duplicate slug in registry"
    assert len(modules) == len(set(modules)), "duplicate module stem in registry"
