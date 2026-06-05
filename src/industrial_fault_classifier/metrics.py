from __future__ import annotations

from collections import Counter

from .constants import TASKS
from .data import Record
from .labels import task_labels


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    total = len(y_true)
    correct = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected == predicted)
    per_label = {}
    precision_values = []
    recall_values = []
    f1_values = []
    for label in labels:
        tp = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected == label and predicted == label)
        fp = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected != label and predicted == label)
        fn = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected == label and predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(1 for item in y_true if item == label),
        }
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_precision": round(sum(precision_values) / len(labels), 6) if labels else 0.0,
        "macro_recall": round(sum(recall_values) / len(labels), 6) if labels else 0.0,
        "macro_f1": round(sum(f1_values) / len(labels), 6) if labels else 0.0,
        "per_label": per_label,
    }


def multitask_metrics(records: list[Record], predictions: list[Record], schema: dict) -> dict:
    metrics = {}
    exact_match = 0
    for expected, predicted in zip(records, predictions, strict=True):
        if all(expected[task] == predicted[task] for task in TASKS):
            exact_match += 1

    for task in TASKS:
        y_true = [record[task] for record in records]
        y_pred = [record[task] for record in predictions]
        metrics[task] = classification_metrics(y_true, y_pred, task_labels(schema, task))

    p01_true = [record["risk_level"] in {"P0", "P1"} for record in records]
    p01_pred = [record["risk_level"] in {"P0", "P1"} for record in predictions]
    tp = sum(1 for expected, predicted in zip(p01_true, p01_pred, strict=True) if expected and predicted)
    fn = sum(1 for expected, predicted in zip(p01_true, p01_pred, strict=True) if expected and not predicted)
    metrics["p0_p1_recall"] = round(tp / (tp + fn), 6) if tp + fn else 0.0
    metrics["three_task_exact_match"] = round(exact_match / len(records), 6) if records else 0.0
    return metrics


def confusion_pairs(records: list[Record], predictions: list[Record], task: str, top_k: int = 20) -> list[dict]:
    pairs = Counter((expected[task], predicted[task]) for expected, predicted in zip(records, predictions, strict=True))
    rows = []
    for (expected, predicted), count in pairs.most_common(top_k):
        rows.append({"expected": expected, "predicted": predicted, "count": count})
    return rows

