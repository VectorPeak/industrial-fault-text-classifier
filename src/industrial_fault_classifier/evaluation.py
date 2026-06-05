"""Model evaluation and batch prediction helpers."""

from __future__ import annotations

from pathlib import Path

from .baseline import MultitaskNaiveBayes
from .config import load_json, write_json
from .constants import TASKS
from .data import Record, read_records, write_records
from .labels import load_label_schema
from .metrics import confusion_pairs, multitask_metrics


def load_backend(model_dir: str | Path) -> str:
    """Read the model backend from metadata, with baseline fallback."""
    metadata_path = Path(model_dir) / "metadata.json"
    if not metadata_path.exists():
        if (Path(model_dir) / "model.pkl").exists():
            return "naive_bayes"
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
    return str(load_json(metadata_path).get("backend", "naive_bayes"))


def predict_records(model_dir: str | Path, records: list[Record]) -> tuple[list[Record], dict]:
    """Load a supported model backend and predict a list of records."""
    model_path = Path(model_dir)
    backend = load_backend(model_path)
    schema = load_label_schema(model_path / "labels.json")
    if backend == "naive_bayes":
        # The unified evaluator currently supports the lightweight baseline path.
        model = MultitaskNaiveBayes.load(model_path / "model.pkl")
        return model.predict_many(records), schema
    raise ValueError("BERT evaluation is intentionally kept separate from the lightweight smoke-test evaluator.")


def evaluate_model(
    model_dir: str | Path,
    data_path: str | Path,
    report_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
    max_samples: int | None = None,
) -> dict:
    """Evaluate a model and optionally write metrics and predictions to disk."""
    records = read_records(data_path)
    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]
    predictions, schema = predict_records(model_dir, records)
    # Core task metrics are augmented with frequent confusion pairs for diagnosis.
    metrics = multitask_metrics(records, predictions, schema)
    metrics["confusion_top"] = {task: confusion_pairs(records, predictions, task) for task in TASKS}

    if report_path:
        write_json(report_path, metrics)
    if prediction_path:
        write_records(prediction_path, predictions)
    return metrics
