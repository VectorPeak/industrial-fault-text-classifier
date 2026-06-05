# step1_dataset_prepare 技术说明
# 报修文本数据标准化与 CSV 固化

## 0x01. 这个文件是做什么的？

Step 1 负责把原始报修文本数据统一转换为项目内部使用的标准 CSV / TSV 结构。它不做模型训练，也不改变标签含义，核心目标是解决数据入口不统一的问题，使后续 EDA、清洗、切分和训练都能基于同一套字段约定运行。

本项目要求每条样本包含 4 个字段：

| 字段 | 含义 | 是否必需 |
|------|------|----------|
| `text` | 单条设备报修文本 | 是 |
| `fault_category` | 故障大类标签 | 是 |
| `risk_level` | 停机风险等级标签 | 是 |
| `department` | 推荐处理部门标签 | 是 |

Step 1 的输出不是最终训练集，而是“标准化后的全量数据文件”。后续 Step 2 会对它做数据探索，Step 3 才会生成 `train / val / test`。

---

## 0x02. 整体流程图

```text
原始 TXT / TSV / CSV 数据
    |
    v
读取文件并判断分隔符
    |
    v
识别是否已有表头
    |
    v
统一字段为 text / fault_category / risk_level / department
    |
    v
基础行级校验
    |
    v
写出标准 CSV / TSV
```

这一步的原则是“只做格式统一，不做业务过滤”。如果在 Step 1 里提前删除样本，后续很难判断问题来自原始数据还是清洗策略。

---

## 0x03. 代码结构说明

Step 1 对应脚本：

```text
scripts/step1_convert_to_csv.py
```

这个脚本本身很薄，只负责把命令转发给统一 CLI：

```python
main(["convert", *sys.argv[1:]])
```

真正的数据读取与写出逻辑位于：

```text
src/industrial_fault_classifier/data.py
src/industrial_fault_classifier/pipeline.py
```

### 3.1 `read_records`

`read_records` 负责读取原始数据。它支持两类输入：

1. 已经带表头的标准 CSV / TSV。
2. 没有表头、但每行正好 4 列的原始 TSV。

如果文件扩展名为 `.csv`，默认使用逗号分隔；否则使用制表符分隔。这样可以兼容公开 CSV、内部 TXT/TSV 导出文件和人工整理后的中间数据。

### 3.2 `convert_dataset`

`convert_dataset` 是 Step 1 的入口函数，执行顺序为：

1. 调用 `read_records` 读取输入数据。
2. 可选截取前 `sample_rows` 行，用于生成公开样例或 smoke test 数据。
3. 调用 `write_records` 写出标准结构。
4. 返回 `validate_records` 的基础校验结果。

### 3.3 `validate_records`

`validate_records` 不删除数据，只统计：

| 指标 | 含义 |
|------|------|
| `rows` | 总行数 |
| `invalid_rows` | 字段结构不完整的行数 |
| `empty_fields` | 关键字段为空的次数 |

这些信息用于判断原始数据是否具备进入 Step 2 / Step 3 的基本条件。

---

## 0x04. 运行命令

从原始 TXT / TSV 转换为标准 CSV：

```powershell
python scripts\step1_convert_to_csv.py `
  --input data\raw\repair_records.txt `
  --output data\full\chemical_repair_text_dataset_cn.csv
```

生成小规模样例数据：

```powershell
python scripts\step1_convert_to_csv.py `
  --input data\full\chemical_repair_text_dataset_cn.csv `
  --output data\samples\sample_repair_text.csv `
  --sample-rows 200
```

如果已经安装项目包，也可以使用统一命令：

```powershell
industrial-fault-classifier convert `
  --input data\raw\repair_records.txt `
  --output data\full\chemical_repair_text_dataset_cn.csv
```

---

## 0x05. 输出产物

| 路径 | 说明 | 是否建议提交 |
|------|------|--------------|
| `data/full/chemical_repair_text_dataset_cn.csv` | 标准化后的公开全量 CSV | 是，当前项目允许提交该公开数据 |
| `data/samples/sample_repair_text.csv` | 用于 smoke test 的小样例 | 是 |
| `data/raw/` | 本地原始数据目录 | 否 |

企业真实工单、内部设备编号、人员信息和未经脱敏的现场文本不应写入公开仓库。

---

## 0x06. 检查点

完成 Step 1 后，应确认以下事项：

1. 输出文件包含 `text / fault_category / risk_level / department` 四列。
2. `text` 字段为原始报修描述，不应混入标签答案。
3. 标签字段使用稳定的中文类别或 `P0 / P1 / P2 / P3` 风险等级。
4. `data/raw/` 不进入 Git 提交范围。
5. 样例数据只用于流程验证，不代表正式模型性能。
