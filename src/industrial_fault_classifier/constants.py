"""数据处理、训练和推理阶段共用的常量定义。
Shared constants used across data processing, training, and inference.
"""

from __future__ import annotations

# 公开 CSV、原始 TSV 转换和训练脚本均使用这四个标准字段。
# Public CSV files, raw-TSV conversion, and training scripts all use this four-column schema.
COLUMNS = ("text", "fault_category", "risk_level", "department")

# 三个监督任务共享同一条报修文本输入，但输出不同标签。
# The three supervised tasks share the same repair text input but predict different labels.
TASKS = ("fault_category", "risk_level", "department")

# 用户未提供 --text 时，CLI 使用这条样例文本完成 smoke prediction。
# Default text for CLI smoke prediction when the user does not provide input.
DEFAULT_TEXT = "空压机运行中压力波动明显，主线节拍受到影响，请安排检修。"
