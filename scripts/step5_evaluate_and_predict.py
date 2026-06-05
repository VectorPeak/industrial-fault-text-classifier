"""Step 5 wrapper: evaluate a model or run single-text prediction."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly without installing the package first.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_fault_classifier.cli import main


if __name__ == "__main__":
    # Step 5 exposes two CLI subcommands, so the first user argument is kept.
    main(sys.argv[1:])
