"""轻量 baseline 后端的单条文本推理封装。
Single-text inference wrapper for the lightweight baseline backend.
"""

from __future__ import annotations

from pathlib import Path

from .baseline import MultitaskNaiveBayes
from .constants import DEFAULT_TEXT
from .data import Record


def predict_text(model_dir: str | Path, text: str = DEFAULT_TEXT) -> Record:
    """加载已保存 baseline，并预测一条报修文本。
    Load a saved baseline model and predict one repair text.
    """
    # 单条推理复用 baseline 的 predict_one，保证 CLI predict 与批量评估逻辑一致。
    # Single-text inference reuses predict_one so CLI predict and batch evaluation stay consistent.
    model = MultitaskNaiveBayes.load(Path(model_dir) / "model.pkl")
    return model.predict_one(text)
