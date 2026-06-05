"""模型评估与批量预测辅助函数。
Model evaluation and batch prediction helpers.
"""

from __future__ import annotations

from pathlib import Path

from .baseline import MultitaskNaiveBayes
from .config import load_json, write_json
from .constants import TASKS
from .data import Record, read_records, write_records
from .labels import load_label_schema
from .metrics import confusion_pairs, multitask_metrics


def load_backend(model_dir: str | Path) -> str:
    """从 metadata 读取模型后端，并兼容旧 baseline 产物。
    Read the model backend from metadata, with baseline fallback.
    """
    metadata_path = Path(model_dir) / "metadata.json"
    if not metadata_path.exists():
        # 早期 baseline 可能只有 model.pkl，因此这里保留向后兼容。
        # Earlier baseline artifacts may only contain model.pkl, so keep backward compatibility.
        if (Path(model_dir) / "model.pkl").exists():
            return "naive_bayes"
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
    return str(load_json(metadata_path).get("backend", "naive_bayes"))


def predict_records(model_dir: str | Path, records: list[Record]) -> tuple[list[Record], dict]:
    """加载受支持模型后端，并对记录列表执行批量预测。
    Load a supported model backend and predict a list of records.
    """
    model_path = Path(model_dir)
    backend = load_backend(model_path)
    schema = load_label_schema(model_path / "labels.json")
    if backend == "naive_bayes":
        # 当前统一评估器支持轻量 baseline；BERT 产物后续可接入同一接口。
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
    """评估模型，并可选写出指标报告和预测结果。
    Evaluate a model and optionally write metrics and predictions to disk.
    """
    records = read_records(data_path)
    if max_samples is not None and max_samples > 0:
        # max_samples 用于快速抽样验证，不修改输入数据。
        # max_samples supports quick sampled evaluation without modifying the input dataset.
        records = records[:max_samples]
    predictions, schema = predict_records(model_dir, records)
    # 在核心指标之外附加高频混淆组合，便于定位典型误分边界。
    # Core task metrics are augmented with frequent confusion pairs for diagnosis.
    metrics = multitask_metrics(records, predictions, schema)
    metrics["confusion_top"] = {task: confusion_pairs(records, predictions, task) for task in TASKS}

    if report_path:
        # JSON 报告用于文档展示、后续可视化和自动化对比。
        # JSON reports are used for documentation, later visualization, and automated comparison.
        write_json(report_path, metrics)
    if prediction_path:
        # 预测 CSV 保留 text 与三任务预测结果，便于抽样复核。
        # Prediction CSV keeps text and three-task outputs for sample review.
        write_records(prediction_path, predictions)
    return metrics
