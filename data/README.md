# 数据目录说明

本目录用于存放报修文本分类项目的数据文件。公开仓库只保留可复现实验流程所需的 CSV 数据；企业真实工单、原始采集文件、清洗后的切分数据和模型训练产物不应提交到 GitHub。

## 0x01. 目录职责

| 路径 | 是否提交 | 说明 |
|------|----------|------|
| `data/full/chemical_repair_text_dataset_cn.csv` | 是 | 基于 Kaggle 公开企业数据集清洗增强后的全量 CSV，约 22 万条样本，用于 Step 2 数据分析和 Step 3 清洗切分 |
| `data/samples/sample_repair_text.csv` | 是 | 小规模公开样例，约 200 条样本，用于快速 smoke test 和命令行流程验证 |
| `data/raw/` | 否 | 本地原始 TXT/TSV 来源文件目录，已被 `.gitignore` 忽略，不上传公开仓库 |
| `data/processed/` | 否 | Step 3 生成的标准化数据、标签映射、训练集、验证集和测试集目录，已被 `.gitignore` 忽略 |

## 0x02. CSV 字段约定

已提交的全量数据和样例数据均采用带表头的标准 CSV 格式，字段如下：

| 字段 | 含义 | 说明 |
|------|------|------|
| `text` | 报修文本 | 单条设备报修、点检或维修描述 |
| `fault_category` | 故障大类 | 例如机械故障、电气故障、传感器故障、控制系统故障等 |
| `risk_level` | 停机风险等级 | `P0 / P1 / P2 / P3`，用于辅助风险优先级判断 |
| `department` | 推荐处理部门 | 用于辅助维修派单和责任部门推荐 |

## 0x03. 使用方式

公开流程默认从全量 CSV 开始运行：

```powershell
python scripts\step2_dataset_eda.py --input data\full\chemical_repair_text_dataset_cn.csv --report artifacts\reports\eda_report.json
python scripts\step3_clean_and_split.py --input data\full\chemical_repair_text_dataset_cn.csv --output-dir data\processed\splits --labels data\processed\labels.json --report artifacts\reports\split_report.json
```

如果只需要快速验证工程闭环，可以使用根目录 README 中的 `industrial-fault-go` 命令。该命令会使用 `data/samples/sample_repair_text.csv` 运行小规模样例流程。

## 0x04. 数据提交规则

1. `data/full/` 和 `data/samples/` 中的公开 CSV 可以随仓库提交。
2. `data/raw/` 仅用于本地原始数据存放，不应提交。
3. `data/processed/` 中的清洗结果、切分数据和调试数据均为生成产物，不应提交。
4. 评估报告、预测结果和模型文件应写入 `artifacts/reports/` 或 `artifacts/models/`，这些目录默认不提交。
5. 如接入真实企业工单，应先完成授权、脱敏和数据安全审查，再决定是否生成公开样例。
