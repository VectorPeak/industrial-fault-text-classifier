"""数据集读写、校验、清洗和分层切分工具。
Dataset IO, validation, cleaning, and stratified splitting utilities.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .constants import COLUMNS, TASKS

Record = dict[str, str]

# 用于识别输入文本中直接暴露标签答案的短语，防止模型学习泄漏特征。
# These patterns identify phrases that directly reveal labels in the input text.
LEAKAGE_PATTERNS = (
    re.compile(r"风险等级[:：]?\s*P[0-3]"),
    re.compile(r"建议按\s*P[0-3]\s*工单处理"),
    re.compile(r"按\s*P[0-3]\s*工单处理"),
    re.compile(r"请尽快安排(?:设备保养/备件管理|工艺/设备工程师|质量/工艺/设备|公辅/设备维修|自动化工程师|自动化/仪表|设备保养|电气维修|机械维修|安全/EHS)"),
)


def _delimiter_for(path: Path) -> str:
    """根据文件扩展名选择 CSV 或 TSV 分隔符。
    Choose CSV or TSV parsing based on file extension.
    """
    if path.suffix.lower() == ".csv":
        return ","
    return "\t"


def read_records(path: str | Path) -> list[Record]:
    """读取带表头 CSV/TSV 或原始四列 TSV，并统一为标准记录。
    Read headered CSV/TSV or raw four-column TSV into normalized records.
    """
    input_path = Path(path)
    delimiter = _delimiter_for(input_path)
    records: list[Record] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        # 先探测第一行是否为标准表头；utf-8-sig 可兼容带 BOM 的 CSV。
        # Probe the first row for the canonical header; utf-8-sig tolerates BOM-prefixed CSV files.
        sample = file.readline()
        file.seek(0)
        first_cells = sample.strip().split(delimiter)
        has_header = tuple(first_cells[:4]) == COLUMNS
        if has_header:
            # 标准数据集已有表头，直接按固定字段读取并去除首尾空白。
            # Standard datasets are headered and use the canonical column names.
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                records.append({column: (row.get(column) or "").strip() for column in COLUMNS})
        else:
            # 原始来源文件默认是无表头四列格式，字段顺序必须与 COLUMNS 一致。
            # Raw source files are expected to contain exactly four columns in COLUMNS order.
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                if not row:
                    continue
                if len(row) != 4:
                    raise ValueError(f"Expected 4 columns in {input_path}, got {len(row)}: {row[:2]}")
                records.append({column: value.strip() for column, value in zip(COLUMNS, row, strict=True)})
    return records


def write_records(path: str | Path, records: list[Record]) -> None:
    """按稳定表头写出记录，并根据后缀选择 CSV/TSV 分隔符。
    Write records with a stable header and delimiter selected by suffix.
    """
    output_path = Path(path)
    # 所有生成数据都通过这一层写出，保证字段顺序和编码一致。
    # All generated datasets pass through this writer to keep column order and encoding consistent.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    delimiter = _delimiter_for(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(records)


def validate_records(records: list[Record]) -> dict[str, int]:
    """统计异常行和空字段数量，但不修改原始数据。
    Count invalid rows and empty fields without mutating the dataset.
    """
    invalid_rows = 0
    empty_fields = 0
    for record in records:
        # 这里只做审计计数；真正删除样本由 clean_records 统一完成。
        # This function only audits counts; clean_records owns destructive filtering.
        if set(record) < set(COLUMNS):
            invalid_rows += 1
        empty_fields += sum(1 for column in COLUMNS if not record.get(column, "").strip())
    return {"rows": len(records), "invalid_rows": invalid_rows, "empty_fields": empty_fields}


def label_distribution(records: list[Record], column: str) -> Counter[str]:
    """统计单个任务字段的标签分布。
    Return per-label counts for one task column.
    """
    return Counter(record[column] for record in records)


def combo_distribution(records: list[Record]) -> Counter[tuple[str, str, str]]:
    """统计三任务组合标签分布。
    Return joint label counts across the three tasks.
    """
    return Counter(tuple(record[task] for task in TASKS) for record in records)


def text_length_summary(records: list[Record]) -> dict[str, float | int]:
    """汇总文本长度，用于辅助设置 tokenizer max_length。
    Summarize repair-text length for tokenizer max_length decisions.
    """
    lengths = sorted(len(record["text"]) for record in records)
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "p50": 0, "p90": 0, "p95": 0}

    def percentile(ratio: float) -> int:
        """按排序后的长度列表取近似分位数。
        Pick an approximate percentile from the sorted length list.
        """
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
    """查找同一文本对应多组三任务标签的冲突样本。
    Find same-text samples that map to multiple label triples.
    """
    labels_by_text: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for record in records:
        # 以完整三任务标签作为监督信号，任何一项不一致都视为冲突。
        # The full three-task tuple is the supervision signal; any mismatch is treated as a conflict.
        labels_by_text[record["text"]].add(tuple(record[task] for task in TASKS))
    return {text: labels for text, labels in labels_by_text.items() if len(labels) > 1}


def count_leakage_phrases(records: list[Record]) -> int:
    """统计包含显式标签泄漏短语的文本数量。
    Count texts that contain direct label leakage phrases.
    """
    count = 0
    for record in records:
        # 一个文本可能命中多个泄漏模式，因此这里累计模式命中次数。
        # One text can match multiple leakage patterns, so this counts pattern hits.
        count += sum(1 for pattern in LEAKAGE_PATTERNS if pattern.search(record["text"]))
    return count


def clean_records(records: list[Record]) -> tuple[list[Record], dict[str, int]]:
    """删除空字段样本、完全重复样本和同文本多标签冲突样本。
    Remove empty rows, exact duplicates, and same-text label conflicts.
    """
    cleaned: list[Record] = []
    exact_seen: set[tuple[str, str, str, str]] = set()
    removed_empty = 0
    removed_exact_duplicates = 0
    for record in records:
        # 先做字段级标准化；文本内部空白折叠可降低换行、制表符带来的特征噪声。
        # Normalize fields first; folding text whitespace reduces newline/tab feature noise.
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

    # 同文本多标签冲突无法判断哪组标签更可信，因此整组剔除以保护监督信号。
    # Conflicting texts are removed entirely because no single label tuple is reliable.
    conflicts = find_conflicting_texts(cleaned)
    if conflicts:
        cleaned = [record for record in cleaned if record["text"] not in conflicts]

    # 清洗统计会写入 split_report，便于追踪数据损耗来源。
    # Cleaning statistics are written to split_report so data loss remains auditable.
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
    """按三任务组合标签分层切分，尽量保持任务相关性。
    Split records by the joint three-task label to preserve task correlation.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    rng = random.Random(seed)
    grouped: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    for record in records:
        # 组合标签比单任务标签更贴近真实派单场景，能减少联合分布偏移。
        # The combined label better matches dispatch usage and reduces joint-distribution drift.
        grouped[tuple(record[task] for task in TASKS)].append(record)

    splits = {"train": [], "val": [], "test": []}
    for group_records in grouped.values():
        # 极小组合优先保留在训练集，避免稀有派单模式完全无法被模型看到。
        # Very small groups are kept in training so rare dispatch combinations are not lost.
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
        # 每个子集内部再次打乱，避免同一组合样本连续出现影响训练顺序。
        # Shuffle each split again so records from the same group do not appear in long contiguous blocks.
        rng.shuffle(split_records)
    return splits
