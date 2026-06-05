"""三任务报修文本分类模型的评估指标。
Evaluation metrics for the three-task repair text classifier.
"""

from __future__ import annotations

from collections import Counter

from .constants import TASKS
from .data import Record
from .labels import task_labels


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """计算单个任务的 accuracy 与 macro precision/recall/F1。
    Compute accuracy and macro-averaged classification metrics for one task.
    """
    total = len(y_true)
    correct = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected == predicted)
    per_label = {}
    precision_values = []
    recall_values = []
    f1_values = []
    for label in labels:
        # 先逐类别计算 TP/FP/FN，再做宏平均，避免高频类别掩盖低频类别表现。
        # Score each label independently before macro averaging so frequent classes do not hide rare-class behavior.
        tp = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected == label and predicted == label)
        fp = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected != label and predicted == label)
        fn = sum(1 for expected, predicted in zip(y_true, y_pred, strict=True) if expected == label and predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        # support 保留每个标签的真实样本量，便于后续判断指标是否受样本稀疏影响。
        # support keeps the true sample count per label, which helps diagnose sparse-label metrics.
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
    """计算三任务指标、三任务全对率和 P0/P1 高风险召回。
    Compute per-task metrics plus joint exact match and P0/P1 recall.
    """
    metrics = {}
    exact_match = 0
    for expected, predicted in zip(records, predictions, strict=True):
        # 三任务全对率要求故障大类、风险等级、处理部门同时预测正确。
        # Joint exact match requires fault category, risk level, and department to be correct together.
        if all(expected[task] == predicted[task] for task in TASKS):
            exact_match += 1

    for task in TASKS:
        # 每个任务仍单独输出完整指标，方便定位是哪一类输出拖累整体派单效果。
        # Each task keeps its own metrics so downstream analysis can locate the weak output dimension.
        y_true = [record[task] for record in records]
        y_pred = [record[task] for record in predictions]
        metrics[task] = classification_metrics(y_true, y_pred, task_labels(schema, task))

    # P0/P1 漏判在化工报修场景中代价更高，因此单独抽取高风险召回。
    # High-risk recall is tracked separately because P0/P1 misses are costly in chemical repair dispatch.
    p01_true = [record["risk_level"] in {"P0", "P1"} for record in records]
    p01_pred = [record["risk_level"] in {"P0", "P1"} for record in predictions]
    tp = sum(1 for expected, predicted in zip(p01_true, p01_pred, strict=True) if expected and predicted)
    fn = sum(1 for expected, predicted in zip(p01_true, p01_pred, strict=True) if expected and not predicted)
    metrics["p0_p1_recall"] = round(tp / (tp + fn), 6) if tp + fn else 0.0
    metrics["three_task_exact_match"] = round(exact_match / len(records), 6) if records else 0.0
    return metrics


def confusion_pairs(records: list[Record], predictions: list[Record], task: str, top_k: int = 20) -> list[dict]:
    """返回最常见的真实/预测标签组合，用于误分边界分析。
    Return the most frequent expected/predicted pairs for error analysis.
    """
    pairs = Counter((expected[task], predicted[task]) for expected, predicted in zip(records, predictions, strict=True))
    rows = []
    for (expected, predicted), count in pairs.most_common(top_k):
        # 保留正确分类与错误分类的组合，便于统一观察主要模式。
        # Keep both correct and incorrect pairs so the report shows dominant patterns consistently.
        rows.append({"expected": expected, "predicted": predicted, "count": count})
    return rows
