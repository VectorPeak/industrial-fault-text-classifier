"""Step 5 入口脚本：评估模型或执行单条报修文本预测。
Step 5 wrapper: evaluate a model or run single-text prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许用户直接运行脚本，而不必先执行 pip install -e .
# Allow direct script execution before installing the local package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_fault_classifier.cli import main


if __name__ == "__main__":
    # Step 5 同时支持 evaluate 和 predict，因此保留用户传入的第一个子命令。
    # Step 5 exposes both "evaluate" and "predict", so the first user subcommand is preserved.
    main(sys.argv[1:])
