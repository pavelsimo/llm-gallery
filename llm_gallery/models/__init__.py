"""Self-contained model implementations — one file per gallery entry.

Each module is independently readable (every block is defined inline) and exposes the same surface:
a ``Config`` dataclass, a ``PRESETS`` dict, and a ``Model``. See ``registry.py`` for the full list and
``llm_gallery/harness/interface.py`` for the contract.
"""
