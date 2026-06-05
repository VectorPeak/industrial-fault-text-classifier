from __future__ import annotations

import shutil
from pathlib import Path

from .baseline import MultitaskNaiveBayes
from .bert import train_bert_classifier
from .config import write_json
from .data import Record, read_records
from .labels import build_label_schema, load_label_schema, save_label_schema


def load_records_with_limit(path: str | Path, max_samples: int | None = None) -> list[Record]:
    records = read_records(path)
    if max_samples is not None and max_samples > 0:
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
    train_records = load_records_with_limit(train_path, max_train_samples)
    val_records = read_records(val_path)
    schema = load_label_schema(labels_path) if Path(labels_path).exists() else build_label_schema(train_records)
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_label_schema(output_dir / "labels.json", schema)

    if backend == "naive_bayes":
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

    raise ValueError(f"Unsupported backend: {backend}")

