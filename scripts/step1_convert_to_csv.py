"""Step 1 入口脚本：将原始报修文本数据标准化为 CSV/TSV。
Step 1 wrapper: standardize raw repair text data into CSV/TSV format.
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
    # 将参数解析与实际执行统一交给共享 CLI，避免脚本层重复实现业务逻辑。
    # Delegate argument parsing and execution to the shared CLI to avoid duplicated logic.
    main(["convert", *sys.argv[1:]])
