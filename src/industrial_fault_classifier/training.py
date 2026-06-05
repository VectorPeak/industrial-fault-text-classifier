"""baseline 与 BERT 后端的训练入口。
Training entry points for baseline and BERT backends.
"""

from __future__ import annotations

from pathlib import Path

from .baseline import MultitaskNaiveBayes
from .bert import train_bert_classifier
from .config import write_json
from .data import Record, read_records
from .labels import build_label_schema, load_label_schema, save_label_schema


def load_records_with_limit(path: str | Path, max_samples: int | None = None) -> list[Record]:
    """读取训练记录，并可选限制样本数以支持快速 smoke run。
    Load records and optionally cap training size for fast smoke runs.
    """
    records = read_records(path)
    if max_samples is not None and max_samples > 0:
        # max_samples 主要服务于演示和 CI，不改变原始数据文件。
        # max_samples is intended for demos and CI; it does not modify the source dataset.
        return records[:max_samples]
    return records


def train_model(
    train_path: str | Path,
    val_path: str | Path,
    labels_path: str | Path,
    model_dir: str | Path,
    backend: str = "naive_bayes",
    max_train_samples: int | None = None,
    alpha: float = 1.0,
    ngram_range: tuple[int, int] = (1, 2),
    bert_model_name: str = "bert-base-chinese",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 96,
) -> None:
    """训练指定后端，并将模型产物写入 model_dir。
    Train the requested backend and write model artifacts to model_dir.
    """
    train_records = load_records_with_limit(train_path, max_train_samples)
    val_records = read_records(val_path)
    # 优先复用 Step 3 生成的标签体系，避免训练、评估和推理阶段标签 ID 不一致。
    # Prefer the Step 3 label schema so training, evaluation, and inference share the same label IDs.
    schema = load_label_schema(labels_path) if Path(labels_path).exists() else build_label_schema(train_records)
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 将标签体系复制到模型目录，使模型产物具备自包含推理能力。
    # Copy the label schema into the model artifact so prediction is self-contained.
    save_label_schema(output_dir / "labels.json", schema)

    if backend == "naive_bayes":
        # baseline 依赖少、训练快，适合验证工程链路和本地无 GPU 环境。
        # The baseline is intentionally dependency-light and suitable for CI checks.
        model = MultitaskNaiveBayes(alpha=alpha, ngram_range=ngram_range).fit(train_records, schema)
        model.save(output_dir / "model.pkl")
        write_json(
            output_dir / "metadata.json",
            {
                "backend": "naive_bayes",
                "train_rows": len(train_records),
                "val_rows": len(val_records),
                "alpha": alpha,
                "ngram_range": list(ngram_range),
            },
        )
        return

    if backend == "bert":
        # BERT 路径执行 encoder 与 task heads 的全量微调，适合正式语义建模实验。
        # BERT training performs full fine-tuning of encoder and task heads.
        train_bert_classifier(
            train_records=train_records,
            val_records=val_records,
            schema=schema,
            model_dir=output_dir,
            model_name=bert_model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
        )
        write_json(
            output_dir / "metadata.json",
            {
                "backend": "bert",
                "train_rows": len(train_records),
                "val_rows": len(val_records),
                "model_name": bert_model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
            },
        )
        return

    # 如果传入未知后端，立即失败，避免静默生成不可识别产物。
    # Fail fast on unknown backends so unsupported artifacts are not produced silently.
    raise ValueError(f"Unsupported backend: {backend}")
