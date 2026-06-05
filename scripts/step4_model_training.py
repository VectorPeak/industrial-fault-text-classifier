"""Step 4 wrapper: train baseline or BERT multi-task classifiers."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly without installing the package first.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_fault_classifier.cli import main


if __name__ == "__main__":
    # Delegate argument parsing and execution to the shared CLI implementation.
    main(["train", *sys.argv[1:]])
