# artifacts 目录说明

`artifacts/` 用于存放项目运行过程中生成的分析报告、预测结果、模型文件和展示图表。该目录的定位是“运行产物目录”，不是源数据目录，也不是正式代码目录。

## 0x01. 目录职责

| 路径 | 内容 | 是否建议提交 |
|------|------|--------------|
| `artifacts/figures/` | README 或技术文档中使用的轻量图表，例如 EDA Dashboard | 可选择提交 |
| `artifacts/reports/` | EDA 报告、切分报告、评估报告、预测明细等 JSON / CSV 文件 | 不建议提交 |
| `artifacts/models/` | baseline、BERT 或后续模型的权重、tokenizer、metadata 等文件 | 不提交 |
| `artifacts/checkpoints/` | 训练中间检查点 | 不提交 |

公开仓库当前只保留经过筛选、适合展示的轻量图表。模型权重、批量预测结果、完整评估报告和训练检查点应在本地生成，不应随仓库发布。

## 0x02. 典型生成产物

Step 2 数据分析：

```text
artifacts/reports/eda_report.json
artifacts/figures/eda_dashboard.png
```

Step 3 清洗与切分：

```text
artifacts/reports/split_report.json
```

Step 4 模型训练：

```text
artifacts/models/baseline/model.pkl
artifacts/models/baseline/labels.json
artifacts/models/baseline/metadata.json
```

Step 5 评估与推理：

```text
artifacts/reports/eval_report.json
artifacts/reports/predictions.csv
```

## 0x03. 提交规则

1. 可提交用于 README 展示的轻量图表，例如 `artifacts/figures/eda_dashboard.png`。
2. 不提交模型权重、训练检查点、完整预测明细和大体积评估产物。
3. 不提交包含企业真实工单、设备编号、人员信息或现场敏感信息的任何文件。
4. 如需复现实验流程，应通过脚本重新生成 reports、models 和 predictions，而不是依赖仓库中的本地运行产物。
