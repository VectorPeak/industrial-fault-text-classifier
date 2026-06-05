from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .constants import COLUMNS, TASKS

Record = dict[str, str]

LEAKAGE_PATTERNS = (
    re.compile(r"风险等级[:：]?\s*P[0-3]"),
    re.compile(r"建议按\s*P[0-3]\s*工单处理"),
    re.compile(r"按\s*P[0-3]\s*工单处理"),
    re.compile(r"请尽快安排(?:设备保养/备件管理|工艺/设备工程师|质量/工艺/设备|公辅/设备维修|自动化工程师|自动化/仪表|设备保养|电气维修|机械维修|安全/EHS)"),
)


def _delimiter_for(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return ","
    return "\t"


def read_records(path: str | Path) -> list[Record]:
    input_path = Path(path)
    delimiter = _delimiter_for(input_path)
    records: list[Record] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.readline()
        file.seek(0)
        first_cells = sample.strip().split(delimiter)
        has_header = tuple(first_cells[:4]) == COLUMNS
        if has_header:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                records.append({column: (row.get(column) or "").strip() for column in COLUMNS})
        else:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                if not row:
                    continue
                if len(row) != 4:
                    raise ValueError(f"Expected 4 columns in {input_path}, got {len(row)}: {row[:2]}")
                records.append({column: value.strip() for column, value in zip(COLUMNS, row, strict=True)})
    return records


def write_records(path: str | Path, records: list[Record]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    delimiter = _delimiter_for(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(records)


def validate_records(records: list[Record]) -> dict[str, int]:
    invalid_rows = 0
    empty_fields = 0
    for record in records:
        if set(record) < set(COLUMNS):
            invalid_rows += 1
        empty_fields += sum(1 for column in COLUMNS if not record.get(column, "").strip())
    return {"rows": len(records), "invalid_rows": invalid_rows, "empty_fields": empty_fields}


def label_distribution(records: list[Record], column: str) -> Counter[str]:
    return Counter(record[column] for record in records)


def combo_distribution(records: list[Record]) -> Counter[tuple[str, str, str]]:
    return Counter(tuple(record[task] for task in TASKS) for record in records)


def text_length_summary(records: list[Record]) -> dict[str, float | int]:
    lengths = sorted(len(record["text"]) for record in records)
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "p50": 0, "p90": 0, "p95": 0}

    def percentile(ratio: float) -> int:
        index = min(len(lengths) - 1, int(round((len(lengths) - 1) * ratio)))
        return lengths[index]

    return {
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(mean(lengths), 4),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
    }


def find_conflicting_texts(records: list[Record]) -> dict[str, set[tuple[str, str, str]]]:
    labels_by_text: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for record in records:
        labels_by_text[record["text"]].add(tuple(record[task] for task in TASKS))
    return {text: labels for text, labels in labels_by_text.items() if len(labels) > 1}


def count_leakage_phrases(records: list[Record]) -> int:
    count = 0
    for record in records:
        count += sum(1 for pattern in LEAKAGE_PATTERNS if pattern.search(record["text"]))
    return count


def clean_records(records: list[Record]) -> tuple[list[Record], dict[str, int]]:
    cleaned: list[Record] = []
    exact_seen: set[tuple[str, str, str, str]] = set()
    removed_empty = 0
    removed_exact_duplicates = 0
    for record in records:
        normalized = {column: record.get(column, "").strip() for column in COLUMNS}
        normalized["text"] = re.sub(r"\s+", "", normalized["text"])
        if any(not normalized[column] for column in COLUMNS):
            removed_empty += 1
            continue
        key = tuple(normalized[column] for column in COLUMNS)
        if key in exact_seen:
            removed_exact_duplicates += 1
            continue
        exact_seen.add(key)
        cleaned.append(normalized)

    conflicts = find_conflicting_texts(cleaned)
    if conflicts:
        cleaned = [record for record in cleaned if record["text"] not in conflicts]

    stats = {
        "input_rows": len(records),
        "cleaned_rows": len(cleaned),
        "removed_empty_rows": removed_empty,
        "removed_exact_duplicates": removed_exact_duplicates,
        "removed_conflicting_text_rows": len(records) - removed_empty - removed_exact_duplicates - len(cleaned),
        "conflicting_text_count": len(conflicts),
    }
    return cleaned, stats


def stratified_split(
    records: list[Record],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[Record]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    rng = random.Random(seed)
    grouped: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[task] for task in TASKS)].append(record)

    splits = {"train": [], "val": [], "test": []}
    for group_records in grouped.values():
        group = list(group_records)
        rng.shuffle(group)
        n = len(group)
        if n == 1:
            train_count = 1
            val_count = 0
        elif n == 2:
            train_count = 1
            val_count = 0
        else:
            train_count = int(n * train_ratio)
            val_count = int(n * val_ratio)
            train_count = max(1, train_count)
            val_count = max(1, val_count)
        if train_count + val_count > n:
            val_count = max(0, n - train_count)
        splits["train"].extend(group[:train_count])
        splits["val"].extend(group[train_count : train_count + val_count])
        splits["test"].extend(group[train_count + val_count :])

    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits
