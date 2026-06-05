"""标签体系构建、持久化和编码解码工具。
Label schema creation, persistence, and encode/decode helpers.
"""

from __future__ import annotations

from pathlib import Path

from .config import load_json, write_json
from .constants import TASKS
from .data import Record


def build_label_schema(records: list[Record]) -> dict:
    """基于当前数据集构建稳定的 label2id / id2label 映射。
    Build deterministic label mappings from the current dataset.
    """
    tasks = {}
    for task in TASKS:
        # 排序后再编号，确保相同数据集在不同机器上生成一致的标签 ID。
        # Sort before indexing so the same dataset receives stable label IDs across machines.
        labels = sorted({record[task] for record in records})
        tasks[task] = {"label2id": {label: index for index, label in enumerate(labels)}}
    return with_id2label({"tasks": tasks})


def with_id2label(schema: dict) -> dict:
    """补齐反向映射，使模型产物可以自描述标签含义。
    Populate reverse mappings so saved artifacts are self-describing.
    """
    for task, payload in schema["tasks"].items():
        label2id = payload["label2id"]
        # JSON 对象键会被保存为字符串，因此 id2label 也显式使用字符串键。
        # JSON object keys are persisted as strings, so id2label uses string keys explicitly.
        payload["id2label"] = {str(index): label for label, index in label2id.items()}
    return schema


def load_label_schema(path: str | Path) -> dict:
    """加载标签体系，并确保反向映射存在。
    Load label schema and ensure reverse mappings exist.
    """
    return with_id2label(load_json(path))


def save_label_schema(path: str | Path, schema: dict) -> None:
    """以便于训练、评估和推理复用的格式保存标签体系。
    Persist label schema in a prediction-friendly format.
    """
    write_json(path, with_id2label(schema))


def task_labels(schema: dict, task: str) -> list[str]:
    """按数值类别 ID 顺序返回某个任务的标签列表。
    Return labels ordered by numeric class id.
    """
    id2label = schema["tasks"][task]["id2label"]
    return [id2label[str(index)] for index in range(len(id2label))]


def encode_label(schema: dict, task: str, label: str) -> int:
    """将人工可读标签转换为模型训练使用的数值 ID。
    Convert a human-readable label to its numeric id.
    """
    return int(schema["tasks"][task]["label2id"][label])


def decode_label(schema: dict, task: str, label_id: int) -> str:
    """将模型输出的数值类别 ID 还原为人工可读标签。
    Convert a numeric class id back to its human-readable label.
    """
    return schema["tasks"][task]["id2label"][str(label_id)]
