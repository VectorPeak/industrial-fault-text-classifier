"""报修文本分类 Pipeline 的命令行接口。
Command line interface for the repair text classification pipeline.
"""

from __future__ import annotations

import argparse
import json

from .config import project_root
from .constants import DEFAULT_TEXT
from .evaluation import evaluate_model
from .inference import predict_text
from .pipeline import build_eda_report, clean_split_dataset, convert_dataset, run_smoke_pipeline
from .training import train_model


def build_parser() -> argparse.ArgumentParser:
    """构建顶层参数解析器及所有子命令。
    Build the top-level parser and all subcommands.
    """
    parser = argparse.ArgumentParser(description="Industrial repair text multi-task classification pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Step 1：将原始四列数据统一为标准 CSV/TSV。
    # Step 1: normalize raw source data.
    convert = subparsers.add_parser("convert", help="Convert raw four-column repair text data to standard CSV/TSV.")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--sample-rows", type=int, default=None)

    # Step 2：检查数据质量、文本长度、标签分布和疑似泄漏。
    # Step 2: inspect dataset quality, text length, label distribution, and possible leakage.
    eda = subparsers.add_parser("eda", help="Generate dataset EDA report.")
    eda.add_argument("--input", required=True)
    eda.add_argument("--report", required=True)

    # Step 3：清洗记录并生成可训练的 train/val/test 切分。
    # Step 3: clean records and create model-ready splits.
    split = subparsers.add_parser("split", help="Clean and stratify dataset.")
    split.add_argument("--input", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--labels", required=True)
    split.add_argument("--report", required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--val-ratio", type=float, default=0.1)
    split.add_argument("--test-ratio", type=float, default=0.1)

    # Step 4：训练用户选择的后端，默认使用轻量 baseline。
    # Step 4: train the selected backend, using the lightweight baseline by default.
    train = subparsers.add_parser("train", help="Train baseline or BERT multi-task classifier.")
    train.add_argument("--train", required=True)
    train.add_argument("--val", required=True)
    train.add_argument("--labels", required=True)
    train.add_argument("--model-dir", required=True)
    train.add_argument("--backend", choices=["naive_bayes", "bert"], default="naive_bayes")
    train.add_argument("--max-train-samples", type=int, default=None)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--model-name", default="bert-base-chinese")
    train.add_argument("--max-length", type=int, default=96)

    # Step 5：在测试集或指定数据集上评估已训练模型。
    # Step 5: evaluate a trained model on a dataset.
    evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    evaluate.add_argument("--model-dir", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--report", default=None)
    evaluate.add_argument("--predictions", default=None)
    evaluate.add_argument("--max-samples", type=int, default=None)

    # Step 5：对单条报修文本执行近似交互式预测。
    # Step 5: run one interactive-style prediction.
    predict = subparsers.add_parser("predict", help="Predict one repair text.")
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--text", default=DEFAULT_TEXT)

    # go 命令用于快速验证公开样例闭环，不依赖 full 数据或 GPU。
    # The go command validates the public-sample loop without requiring full data or GPU.
    subparsers.add_parser("go", help="Run a lightweight end-to-end smoke pipeline on the public sample.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """解析命令行参数，并分发到共享 Pipeline 函数。
    Dispatch parsed CLI commands to the shared pipeline functions.
    """
    args = build_parser().parse_args(argv)
    if args.command == "convert":
        result = convert_dataset(args.input, args.output, args.sample_rows)
    elif args.command == "eda":
        result = build_eda_report(args.input, args.report)
    elif args.command == "split":
        result = clean_split_dataset(
            input_path=args.input,
            output_dir=args.output_dir,
            labels_path=args.labels,
            report_path=args.report,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    elif args.command == "train":
        train_model(
            train_path=args.train,
            val_path=args.val,
            labels_path=args.labels,
            model_dir=args.model_dir,
            backend=args.backend,
            max_train_samples=args.max_train_samples,
            bert_model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
        )
        result = {"model_dir": args.model_dir, "backend": args.backend}
    elif args.command == "evaluate":
        result = evaluate_model(args.model_dir, args.data, args.report, args.predictions, args.max_samples)
    elif args.command == "predict":
        result = predict_text(args.model_dir, args.text)
    elif args.command == "go":
        result = run_smoke_pipeline(project_root())
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    # CLI 统一输出 JSON，便于命令行阅读，也便于其他脚本解析。
    # The CLI always prints JSON so humans can inspect it and scripts can parse it.
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main_go() -> None:
    """运行注册为 industrial-fault-go 的公开样例 smoke pipeline。
    Run the public-sample smoke pipeline registered as industrial-fault-go.
    """
    result = run_smoke_pipeline(project_root())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
