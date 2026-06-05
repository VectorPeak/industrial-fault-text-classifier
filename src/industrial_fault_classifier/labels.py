from __future__ import annotations

from pathlib import Path

from .config import load_json, write_json
from .constants import TASKS
from .data import Record


def build_label_schema(records: list[Record]) -> dict:
    tasks = {}
    for task in TASKS:
        labels = sorted({record[task] for record in records})
        tasks[task] = {"label2id": {label: index for index, label in enumerate(labels)}}
    return with_id2label({"tasks": tasks})


def with_id2label(schema: dict) -> dict:
    for task, payload in schema["tasks"].items():
        label2id = payload["label2id"]
        payload["id2label"] = {str(index): label for label, index in label2id.items()}
    return schema


def load_label_schema(path: str | Path) -> dict:
    return with_id2label(load_json(path))


def save_label_schema(path: str | Path, schema: dict) -> None:
    write_json(path, with_id2label(schema))


def task_labels(schema: dict, task: str) -> list[str]:
    id2label = schema["tasks"][task]["id2label"]
    return [id2label[str(index)] for index in range(len(id2label))]


def encode_label(schema: dict, task: str, label: str) -> int:
    return int(schema["tasks"][task]["label2id"][label])


def decode_label(schema: dict, task: str, label_id: int) -> str:
    return schema["tasks"][task]["id2label"][str(label_id)]

