"""Shared constants used across data processing, training, and inference."""

from __future__ import annotations

# All public datasets and scripts use the same four-column schema.
COLUMNS = ("text", "fault_category", "risk_level", "department")
TASKS = ("fault_category", "risk_level", "department")

# Default text for CLI smoke prediction when the user does not provide input.
DEFAULT_TEXT = "空压机运行中压力波动明显，主线节拍受到影响，请安排检修。"
