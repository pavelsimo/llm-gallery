"""Shared tooling for driving any model: dataset, training loop, sampler, and the model contract.

These modules are deliberately *not* imported by the model files — each model in ``llm_gallery/models``
stays fully self-contained. The harness only depends on the small interface documented in ``interface.py``.
"""
