"""Step 3 入口脚本：清洗数据并生成分层 train/val/test 切分。
Step 3 wrapper: clean data and create stratified train/val/test splits.
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
    # Step 3 的清洗、标签映射和分层切分都由 CLI 的 split 子命令统一处理。
    # The shared "split" subcommand owns cleaning, label-schema creation, and stratified splitting.
    main(["split", *sys.argv[1:]])
