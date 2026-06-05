"""CLI 命令和分步骤脚本共用的端到端 Pipeline 函数。
End-to-end pipeline functions used by CLI commands and step scripts.
"""

from __future__ import annotations

from pathlib import Path

from .config import write_json
from .constants import TASKS
from .data import (
    clean_records,
    combo_distribution,
    count_leakage_phrases,
    find_conflicting_texts,
    label_distribution,
    read_records,
    stratified_split,
    text_length_summary,
    validate_records,
    write_records,
)
from .evaluation import evaluate_model
from .inference import predict_text
from .labels import build_label_schema, save_label_schema
from .training import train_model


def convert_dataset(input_path: str | Path, output_path: str | Path, sample_rows: int | None = None) -> dict:
    """将原始或标准输入转换为统一记录格式并写出到磁盘。
    Convert raw or standard input into canonical records written to disk.
    """
    records = read_records(input_path)
    if sample_rows is not None and sample_rows > 0:
        # 抽样输出用于调试或生成公开样例，不影响原始输入文件。
        # Sampled output is useful for debugging or public samples and does not affect the input file.
        records = records[:sample_rows]
    write_records(output_path, records)
    # 转换后立即返回校验摘要，让调用方知道行数和空字段情况。
    # Return validation immediately after conversion so callers see row and empty-field counts.
    return validate_records(records)


def build_eda_report(input_path: str | Path, report_path: str | Path) -> dict:
    """生成标签、文本长度与泄漏风险诊断的 JSON EDA 报告。
    Build a JSON EDA report for label, text, and leakage diagnostics.
    """
    records = read_records(input_path)
    report = {
        "validation": validate_records(records),
        "text_length": text_length_summary(records),
        # duplicate_text_rows 关注文本重复，不要求三任务标签完全一致。
        # duplicate_text_rows focuses on repeated text regardless of whether labels are identical.
        "duplicate_text_rows": len(records) - len({record["text"] for record in records}),
        "conflicting_text_count": len(find_conflicting_texts(records)),
        "leakage_phrase_hits": count_leakage_phrases(records),
        # 分布统计保持机器可读，方便后续 EDA 图表或报告复用。
        # Keep distributions machine-readable so dashboards can reuse the report.
        "label_distribution": {
            task: dict(label_distribution(records, task).most_common())
            for task in TASKS
        },
        "top_label_combinations": [
            {"fault_category": key[0], "risk_level": key[1], "department": key[2], "count": count}
            for key, count in combo_distribution(records).most_common(30)
        ],
    }
    write_json(report_path, report)
    return report


def clean_split_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    labels_path: str | Path,
    report_path: str | Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """清洗记录、保存标签体系并写出分层数据切分。
    Clean records, save label schema, and write stratified data splits.
    """
    records = read_records(input_path)
    cleaned, clean_stats = clean_records(records)
    # 标签体系必须基于清洗后的数据构建，避免冲突样本污染类别集合。
    # Build the label schema from cleaned data so conflicting samples do not pollute label sets.
    schema = build_label_schema(cleaned)
    save_label_schema(labels_path, schema)
    splits = stratified_split(cleaned, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    output = Path(output_dir)
    paths = {
        "train": output / "train.csv",
        "val": output / "val.csv",
        "test": output / "test.csv",
    }
    for split_name, split_records in splits.items():
        # 切分文件属于生成产物，应写入被忽略的 data/processed 目录。
        # Split files are generated artifacts and stay under ignored directories.
        write_records(paths[split_name], split_records)

    report = {
        "cleaning": clean_stats,
        "split_sizes": {split_name: len(split_records) for split_name, split_records in splits.items()},
        "paths": {split_name: str(path) for split_name, path in paths.items()},
        "label_schema_path": str(labels_path),
    }
    write_json(report_path, report)
    return report


def run_smoke_pipeline(root: str | Path) -> dict:
    """在已提交样例数据上运行轻量端到端 smoke pipeline。
    Run a lightweight end-to-end pipeline on the committed sample dataset.
    """
    root_path = Path(root)
    sample_path = root_path / "data" / "samples" / "sample_repair_text.csv"
    split_dir = root_path / "data" / "processed" / "go_splits"
    labels_path = root_path / "data" / "processed" / "go_labels.json"
    report_dir = root_path / "artifacts" / "reports"
    model_dir = root_path / "artifacts" / "models" / "go_baseline"
    report_dir.mkdir(parents=True, exist_ok=True)

    # smoke 路径只验证工程连接和文件契约，不触碰 full 数据，也不要求 GPU。
    # The smoke path proves wiring and file contracts without touching full data or requiring GPU.
    split_report = clean_split_dataset(
        input_path=sample_path,
        output_dir=split_dir,
        labels_path=labels_path,
        report_path=report_dir / "go_split_report.json",
    )
    # go 流程固定使用 baseline，保证新环境中也能快速跑通。
    # The go workflow uses the baseline so it can run quickly in a fresh environment.
    train_model(
        train_path=split_dir / "train.csv",
        val_path=split_dir / "val.csv",
        labels_path=labels_path,
        model_dir=model_dir,
        backend="naive_bayes",
        max_train_samples=160,
    )
    # 评估报告和预测样例都写入 artifacts/reports，默认不提交到公开仓库。
    # Evaluation reports and predictions go to artifacts/reports, which is ignored by default.
    metrics = evaluate_model(
        model_dir=model_dir,
        data_path=split_dir / "test.csv",
        report_path=report_dir / "go_eval_report.json",
        prediction_path=report_dir / "go_predictions.csv",
    )
    # 返回一条单样本预测，方便用户快速确认推理接口可用。
    # Return one single-text prediction so users can quickly confirm the inference interface.
    prediction = predict_text(model_dir)
    return {"split_report": split_report, "metrics": metrics, "prediction": prediction}
