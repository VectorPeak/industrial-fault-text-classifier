"""Step 2 入口脚本：生成数据探索分析与质量审计报告。
Step 2 wrapper: generate dataset EDA and quality-audit reports.
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
    # 将 Step 2 固定映射到 CLI 的 eda 子命令，保持脚本和包入口行为一致。
    # Map Step 2 to the shared "eda" subcommand so script and package entrypoints stay aligned.
    main(["eda", *sys.argv[1:]])
