"""Step 4 入口脚本：训练 baseline 或 BERT 多任务分类模型。
Step 4 wrapper: train baseline or BERT multi-task classifiers.
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
    # 训练后端通过 --backend 选择，脚本只负责转发到统一训练入口。
    # The training backend is selected by --backend; this script only forwards to the shared entrypoint.
    main(["train", *sys.argv[1:]])
