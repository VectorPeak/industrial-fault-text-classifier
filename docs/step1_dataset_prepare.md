# Step 1 Dataset Prepare 技术说明

## 0x01. 目标

将原始四列报修文本数据标准化为带表头的数据文件，作为后续 EDA、清洗切分和模型训练的统一输入。

## 0x02. 输入与输出

| 项目 | 路径示例 | 说明 |
|------|----------|------|
| 输入 | `data/raw/manufacturing_repair_text_dataset_cn.txt` | 本地原始 TXT/TSV 来源文件，默认不提交 |
| 输出 | `data/full/chemical_repair_text_dataset_cn.csv` | 带表头全量 CSV，提交到公开仓库 |
| 样例 | `data/samples/sample_repair_text.csv` | 公开样例数据 |

标准字段为 `text`、`fault_category`、`risk_level`、`department`。

## 0x03. 核心设计

原始数据通常以 TXT/TSV 形式保存，字段之间使用制表符分隔。标准化阶段会校验每行字段数量、空字段和文件编码，并输出带表头 CSV。CSV 写入使用标准 quoting 规则，可以安全处理中文逗号、报警码和部门名称中的斜杠。

## 0x04. 运行命令

仓库已经提交转换后的全量 CSV。如需从本地原始 TXT/TSV 重新生成，可运行：

```powershell
python scripts\step1_convert_to_csv.py --input data\raw\manufacturing_repair_text_dataset_cn.txt --output data\full\chemical_repair_text_dataset_cn.csv
```

只抽取前 N 行用于调试时，建议写入 `data/processed/`：

```powershell
python scripts\step1_convert_to_csv.py --input data\raw\manufacturing_repair_text_dataset_cn.txt --output data\processed\debug_dataset.csv --sample-rows 1000
```

## 0x05. 产物

全量 CSV 位于 `data/full/chemical_repair_text_dataset_cn.csv`，可直接用于 Step 2 和 Step 3。`data/processed/` 仍用于调试数据、切分结果和其他本地生成产物，不上传到公开仓库。
