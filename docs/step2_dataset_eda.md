# step2_dataset_eda 技术说明
# 多任务文本数据 EDA 与质量审计

## 0x01. 这个文件是做什么的？

Step 2 用于分析标准化后的报修文本数据，重点不是画图本身，而是确认这批数据是否适合训练三任务分类模型。它会统计文本长度、标签分布、组合标签分布、重复文本、同文本多标签冲突以及疑似标签泄漏短语，为 Step 3 的清洗策略提供依据。

在多任务分类项目中，EDA 至少要回答三个问题：

1. 三个任务的标签分布是否严重失衡。
2. 同一条报修文本是否存在互相冲突的标签。
3. 文本中是否直接出现风险等级或处理部门等标签答案。

---

## 0x02. 整体流程图

```text
标准化 CSV 数据
    |
    v
基础字段校验
    |
    +-- 文本长度统计
    |
    +-- fault_category 标签分布
    |
    +-- risk_level 标签分布
    |
    +-- department 标签分布
    |
    +-- 三任务组合标签分布
    |
    +-- 重复文本与冲突标签检查
    |
    +-- 疑似标签泄漏短语统计
    |
    v
输出 EDA JSON 报告
```

README 中展示的 EDA Dashboard 是对 Step 2 报告和公开数据分布的可视化整理，用于快速说明数据结构。

---

## 0x03. 关键统计项

### 3.1 文本长度分布

文本长度直接影响 BERT 的 `max_length` 设置，也会影响字符 n-gram baseline 的稀疏程度。Step 2 统计以下指标：

| 指标 | 含义 |
|------|------|
| `min` | 最短文本长度 |
| `max` | 最长文本长度 |
| `mean` | 平均文本长度 |
| `p50` | 中位数长度 |
| `p90` | 90 分位长度 |
| `p95` | 95 分位长度 |

如果 `p95` 明显低于 96，则 BERT 训练中使用 `max_length=96` 通常可以覆盖大部分样本，同时避免过多无效 padding。

### 3.2 单任务标签分布

项目包含三个监督任务：

```text
fault_category / risk_level / department
```

Step 2 会分别统计每个任务的类别频次。该统计用于判断：

1. 是否存在极少样本类别。
2. 是否需要使用 macro-F1 作为核心指标。
3. 是否需要在训练阶段考虑类别权重或采样策略。

### 3.3 三任务组合标签分布

单独看某个任务的分布并不足够，因为真实派单结果由三类标签共同决定。项目使用组合键：

```text
fault_category + risk_level + department
```

统计高频组合后，可以判断哪些“故障类型-风险等级-处理部门”的组合最常见，也能为 Step 3 的分层切分提供依据。

### 3.4 重复文本与标签冲突

重复数据分为两类：

| 类型 | 处理建议 |
|------|----------|
| 完全重复行 | Step 3 保留一条，删除重复副本 |
| 同一文本对应多组标签 | Step 3 删除冲突文本对应的全部样本 |

第二类问题比第一类更严重，因为它表示监督信号本身不一致。若直接进入训练，模型会被要求把同一句话同时学成不同答案。

### 3.5 疑似标签泄漏

项目会统计类似以下内容：

```text
风险等级: P1
建议按 P0 工单处理
请尽快安排自动化工程师
```

这些短语可能把标签答案直接写入输入文本。它们不一定全部需要删除，但必须被记录和审计，否则模型指标可能来自“读答案”，而不是理解故障语义。

---

## 0x04. 运行命令

生成 EDA 报告：

```powershell
python scripts\step2_dataset_eda.py `
  --input data\full\chemical_repair_text_dataset_cn.csv `
  --report artifacts\reports\eda_report.json
```

使用统一 CLI：

```powershell
industrial-fault-classifier eda `
  --input data\full\chemical_repair_text_dataset_cn.csv `
  --report artifacts\reports\eda_report.json
```

---

## 0x05. 输出产物

| 路径 | 说明 |
|------|------|
| `artifacts/reports/eda_report.json` | 文本长度、标签分布、组合标签和质量审计结果 |
| `artifacts/figures/eda_dashboard.png` | README 中展示的 EDA 组合图 |

`artifacts/reports/` 默认属于运行产物，不建议提交。经过筛选的轻量图像可以提交，用于 README 展示。

---

## 0x06. 检查点

完成 Step 2 后，应重点检查：

1. 三个任务是否存在极端长尾类别。
2. `risk_level` 是否被少数类别支配。
3. `department × fault_category` 是否存在明显的一对多关系。
4. 同文本多标签冲突数量是否可接受。
5. 疑似标签泄漏是否会影响离线评估可信度。
