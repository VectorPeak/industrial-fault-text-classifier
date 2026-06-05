# Step 2 Dataset EDA 技术说明

## 0x01. 目标

对标准化后的报修文本数据进行结构化检查，确认数据规模、标签分布、文本长度和潜在泄漏风险。

## 0x02. 输入与输出

| 项目 | 路径示例 | 说明 |
|------|----------|------|
| 输入 | `data/processed/standard_dataset.csv` | Step 1 输出 |
| 输出 | `artifacts/reports/eda_report.json` | 数据探索报告 |

## 0x03. 核心检查项

| 检查项 | 用途 |
|--------|------|
| 标签分布 | 判断故障大类、风险等级、处理部门是否严重不均衡 |
| 组合标签分布 | 评估三任务联合分层切分是否可行 |
| 文本长度统计 | 为 BERT `max_length` 选择提供依据 |
| 重复文本数量 | 识别 train/val/test 潜在泄漏来源 |
| 疑似泄漏短语 | 检查文本中是否直接出现风险等级或处理部门答案 |

## 0x04. 运行命令

```powershell
python scripts\step2_dataset_eda.py --input data\processed\standard_dataset.csv --report artifacts\reports\eda_report.json
```

## 0x05. 产物

EDA 报告采用 JSON 格式，方便后续脚本或可视化工具继续读取。报告目录默认不提交。

