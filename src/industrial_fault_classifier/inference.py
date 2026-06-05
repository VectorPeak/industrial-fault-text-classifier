"""Single-text inference wrapper for the lightweight baseline backend."""

from __future__ import annotations

from pathlib import Path

from .baseline import MultitaskNaiveBayes
from .constants import DEFAULT_TEXT
from .data import Record


def predict_text(model_dir: str | Path, text: str = DEFAULT_TEXT) -> Record:
    """Load a saved baseline model and predict one repair text."""
    model = MultitaskNaiveBayes.load(Path(model_dir) / "model.pkl")
    return model.predict_one(text)
