"""End-to-end pipeline functions used by CLI commands and step scripts."""

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
    """Convert raw or standard input into canonical records written to disk."""
    records = read_records(input_path)
    if sample_rows is not None and sample_rows > 0:
        records = records[:sample_rows]
    write_records(output_path, records)
    return validate_records(records)


def build_eda_report(input_path: str | Path, report_path: str | Path) -> dict:
    """Build a JSON EDA report for label, text, and leakage diagnostics."""
    records = read_records(input_path)
    report = {
        "validation": validate_records(records),
        "text_length": text_length_summary(records),
        "duplicate_text_rows": len(records) - len({record["text"] for record in records}),
        "conflicting_text_count": len(find_conflicting_texts(records)),
        "leakage_phrase_hits": count_leakage_phrases(records),
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
    """Clean records, save label schema, and write stratified data splits."""
    records = read_records(input_path)
    cleaned, clean_stats = clean_records(records)
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
    """Run a lightweight end-to-end pipeline on the committed sample dataset."""
    root_path = Path(root)
    sample_path = root_path / "data" / "samples" / "sample_repair_text.csv"
    split_dir = root_path / "data" / "processed" / "go_splits"
    labels_path = root_path / "data" / "processed" / "go_labels.json"
    report_dir = root_path / "artifacts" / "reports"
    model_dir = root_path / "artifacts" / "models" / "go_baseline"
    report_dir.mkdir(parents=True, exist_ok=True)

    # The smoke path proves wiring and file contracts without touching full data.
    split_report = clean_split_dataset(
        input_path=sample_path,
        output_dir=split_dir,
        labels_path=labels_path,
        report_path=report_dir / "go_split_report.json",
    )
    train_model(
        train_path=split_dir / "train.csv",
        val_path=split_dir / "val.csv",
        labels_path=labels_path,
        model_dir=model_dir,
        backend="naive_bayes",
        max_train_samples=160,
    )
    metrics = evaluate_model(
        model_dir=model_dir,
        data_path=split_dir / "test.csv",
        report_path=report_dir / "go_eval_report.json",
        prediction_path=report_dir / "go_predictions.csv",
    )
    prediction = predict_text(model_dir)
    return {"split_report": split_report, "metrics": metrics, "prediction": prediction}
