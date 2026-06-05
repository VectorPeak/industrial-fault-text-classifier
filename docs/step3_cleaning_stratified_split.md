# Step 3 Cleaning And Stratified Split 技术说明

## 0x01. 目标

完成训练前的数据清洗、标签映射生成和 `train / val / test` 切分，降低重复样本、冲突标签和切分泄漏对模型评估的影响。

## 0x02. 输入与输出

| 项目 | 路径示例 | 说明 |
|------|----------|------|
| 输入 | `data/full/chemical_repair_text_dataset_cn.csv` | 已提交的全量 CSV |
| 输出数据 | `data/processed/splits/*.csv` | 训练、验证、测试切分 |
| 标签映射 | `data/processed/labels.json` | 当前数据集的 label schema |
| 报告 | `artifacts/reports/split_report.json` | 清洗与切分统计 |

## 0x03. 核心设计

清洗阶段执行以下规则：

1. 删除空字段样本。
2. 删除完全重复行。
3. 剔除同一文本对应多组标签的冲突样本。
4. 基于 `fault_category + risk_level + department` 组合标签进行分层切分。

组合标签分层可以尽量保持三任务联合分布一致，避免某些任务在验证集或测试集中缺少关键类别。

## 0x04. 运行命令

```powershell
python scripts\step3_clean_and_split.py --input data\full\chemical_repair_text_dataset_cn.csv --output-dir data\processed\splits --labels data\processed\labels.json --report artifacts\reports\split_report.json
```

## 0x05. 产物

切分数据和标签映射会写入 `data/processed/`。模型训练时应使用同一份 `labels.json`，避免训练、评估和预测阶段标签 ID 不一致。
