"""项目路径解析与 JSON 配置读写工具。
Project path and JSON configuration helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """从当前源码文件位置推断仓库根目录。
    Return the repository root inferred from the installed source tree.
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """将用户输入路径解析为绝对路径，必要时按项目根目录补全。
    Resolve a user-provided path relative to the project root when needed.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root or project_root()) / candidate


def load_json(path: str | Path) -> dict[str, Any]:
    """读取 UTF-8 编码 JSON 文件。
    Load a UTF-8 JSON file.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """写入 UTF-8 JSON 文件，并自动创建父目录。
    Write a UTF-8 JSON file and create parent directories automatically.
    """
    output_path = Path(path)
    # 报告、标签映射和模型元数据常写入生成目录，因此这里集中处理目录创建。
    # Reports, label schemas, and model metadata are generated files, so parent creation is centralized here.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
