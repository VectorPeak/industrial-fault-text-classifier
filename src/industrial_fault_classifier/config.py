"""Project path and JSON configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root inferred from the installed source tree."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a user-provided path relative to the project root when needed."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root or project_root()) / candidate


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON file and create parent directories automatically."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
