"""Tiny character-level dataset (Karpathy's tiny-shakespeare) for the training demo.

Character-level means *no tokenizer*: every distinct character is a token (vocab ~65). That keeps the
entire data pipeline transparent and lets any architecture train in minutes on CPU. The file (~1MB) is
downloaded once into ``data/``; if there's no network, a small bundled excerpt is used so the demo
still runs.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

# data/ sits at the repo root: harness/ -> llm_gallery/ -> repo root
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
SHAKESPEARE_PATH = DATA_DIR / "tinyshakespeare.txt"

# Fallback so training works fully offline (used only if the download fails).
_FALLBACK = (
    "ROMEO:\nBut soft, what light through yonder window breaks?\n"
    "It is the east, and Juliet is the sun.\n"
    "Arise, fair sun, and kill the envious moon,\n"
    "Who is already sick and pale with grief.\n\n"
    "JULIET:\nO Romeo, Romeo, wherefore art thou Romeo?\n"
    "Deny thy father and refuse thy name;\n"
    "Or if thou wilt not, be but sworn my love,\n"
    "And I'll no longer be a Capulet.\n\n"
    "HAMLET:\nTo be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles.\n\n"
) * 40


@dataclass
class CharDataset:
    """Encoded train/val splits plus the char<->id maps needed to decode samples."""

    train: torch.Tensor  # 1-D LongTensor of token ids
    val: torch.Tensor
    stoi: dict[str, int]  # char -> id
    itos: dict[int, str]  # id -> char

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def _load_text() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SHAKESPEARE_PATH.exists():
        try:
            urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)
        except Exception as exc:  # offline: fall back to the bundled excerpt
            print(f"[data] download failed ({exc}); using bundled fallback text")
            return _FALLBACK
    return SHAKESPEARE_PATH.read_text(encoding="utf-8")


def load_char_shakespeare(val_fraction: float = 0.1) -> CharDataset:
    text = _load_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = dict(enumerate(chars))
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    return CharDataset(train=data[:n], val=data[n:], stoi=stoi, itos=itos)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (inputs, targets); targets are inputs shifted right by one position."""
    high = len(data) - block_size - 1
    ix = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
