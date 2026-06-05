# step3_cleaning_stratified_split 技术说明
# 数据清洗与三任务组合分层切分

## 0x01. 这个文件是做什么的？

Step 3 将经过 Step 1 标准化、Step 2 审计的数据转换为模型训练可用的数据集。它完成三件事：

1. 删除关键字段为空、完全重复和同文本多标签冲突的样本。
2. 生成三任务标签映射文件。
3. 按组合标签进行 `train / val / test` 分层切分。

这一步是整个文本分类 Pipeline 的质量边界。模型训练效果是否可信，首先取决于 Step 3 是否把明显不可靠的数据处理干净。

---

## 0x02. 整体流程图

```text
标准化全量 CSV
    |
    v
字段标准化与空值检查
    |
    v
删除完全重复行
    |
    v
识别同文本多标签冲突
    |
    v
剔除冲突文本对应样本
    |
    v
构建 labels.json
    |
    v
按 fault_category + risk_level + department 分组
    |
    v
组内随机打散并切分
    |
    v
输出 train.csv / val.csv / test.csv
```

---

## 0x03. 核心设计

### 3.1 为什么不只按单任务分层？

本项目不是单一分类任务，而是同时预测：

```text
故障大类 / 停机风险等级 / 处理部门
```

如果只按 `fault_category` 分层，可能出现训练集与测试集在 `risk_level` 或 `department` 上分布不一致的问题。对于派单场景来说，三任务组合才更接近真实业务样本。

因此 Step 3 使用组合键：

```text
(fault_category, risk_level, department)
```

同一组合内部再按比例切分，尽量保持三个子集中的联合分布稳定。

### 3.2 清洗规则

| 清洗项 | 处理方式 | 原因 |
|--------|----------|------|
| 关键字段为空 | 删除该行 | 无法形成完整监督信号 |
| 完全重复行 | 保留首条，删除重复副本 | 避免局部模板被重复放大 |
| 文本空白字符 | 去除换行、制表符和空白 | 降低无意义格式噪声 |
| 同文本多标签冲突 | 删除该文本对应全部样本 | 标签口径不一致，无法作为可靠监督 |

这里不对标签做自动合并，也不尝试用多数投票修复冲突标签。原因是维修派单标签往往包含业务判断，简单投票可能把错误标签固化进训练集。

### 3.3 标签映射

Step 3 会根据清洗后的数据生成标签结构：

```text
labels.json
    |
    +-- fault_category.label2id
    +-- risk_level.label2id
    +-- department.label2id
```

模型训练、评估和推理必须使用同一份标签映射，否则预测结果的 ID 与中文标签会发生错位。

### 3.4 小样本组合处理

在组合分层中，某些组合可能只有 1 条或 2 条样本。代码会优先把极小组合保留在训练集中，避免罕见组合完全丢失。这样做的代价是验证集或测试集中未必覆盖所有组合，但能保证训练阶段至少见过这些稀有标签组合。

---

## 0x04. 运行命令

```powershell
python scripts\step3_clean_and_split.py `
  --input data\full\chemical_repair_text_dataset_cn.csv `
  --output-dir data\processed\splits `
  --labels data\processed\labels.json `
  --report artifacts\reports\split_report.json `
  --seed 42 `
  --train-ratio 0.8 `
  --val-ratio 0.1 `
  --test-ratio 0.1
```

使用统一 CLI：

```powershell
industrial-fault-classifier split `
  --input data\full\chemical_repair_text_dataset_cn.csv `
  --output-dir data\processed\splits `
  --labels data\processed\labels.json `
  --report artifacts\reports\split_report.json
```

---

## 0x05. 输出产物

| 路径 | 说明 | 是否提交 |
|------|------|----------|
| `data/processed/splits/train.csv` | 训练集 | 否 |
| `data/processed/splits/val.csv` | 验证集 | 否 |
| `data/processed/splits/test.csv` | 测试集 | 否 |
| `data/processed/labels.json` | 本次切分对应的标签映射 | 否 |
| `artifacts/reports/split_report.json` | 清洗统计与切分规模 | 否 |

`data/processed/` 是生成产物目录，已加入 `.gitignore`。公开仓库保留全量公开 CSV 与小样例即可。

---

## 0x06. 检查点

完成 Step 3 后，应确认：

1. 清洗前后样本数变化合理。
2. `train / val / test` 比例接近 8:1:1。
3. 标签映射文件与训练数据来自同一次切分。
4. 冲突样本数量没有异常升高。
5. 后续训练命令使用的 `--labels` 路径与 Step 3 输出一致。
