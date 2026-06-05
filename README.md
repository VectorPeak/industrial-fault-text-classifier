<div align="center">

# Industrial Fault Text Classifier | 制造业设备报修文本多任务分类系统

面向制造业报修文本的故障大类、停机风险等级与处理部门多任务分类方案

![python](https://img.shields.io/badge/python-3.10+-3776AB)
![pytorch](https://img.shields.io/badge/PyTorch-optional--BERT-EE4C2C)
![transformers](https://img.shields.io/badge/Transformers-BERT-blue)
![multitask](https://img.shields.io/badge/multi--task-learning-5C7CFA)
![maintenance](https://img.shields.io/badge/industrial-maintenance-green)

简体中文 | [English](README_EN.md)

</div>

```text
repair text -> label mapping -> data quality check -> stratified split -> multi-task classifier -> evaluation -> inference demo
```

> 本仓库公开数据为用于复现实验流程的构造/样例数据，不包含企业真实生产工单、设备编号、人员信息或现场敏感信息。如项目在实际企业场景中落地，真实报修记录及中间数据产物应按企业数据安全要求处理，不在公开仓库中发布。

## 0x01. 项目背景

制造业现场的报修记录、点检记录和维修工单通常以自然语言文本形式存在，文本中同时包含设备、部件、故障现象、风险程度和建议处理方向。传统处理方式依赖调度人员阅读文本后手工分派，容易受到表达不规范、经验差异和高风险工单堆积的影响。

本项目将单条报修文本转换为三个结构化判断结果：

| 任务 | 输出 | 用途 |
|------|------|------|
| 故障大类分类 | 机械故障、电气故障、传感器故障等 | 形成维修统计与故障画像 |
| 停机风险等级识别 | P0 / P1 / P2 / P3 | 辅助识别高优先级工单 |
| 处理部门推荐 | 机械维修、电气维修、自动化工程师等 | 降低转派和等待成本 |

项目定位不是替代工程师判断，而是将非结构化报修文本转化为可检索、可统计、可辅助派单的结构化信息。

---

## 0x02. 数据集与标签体系

当前公开仓库只提交小规模样例数据：

```text
data/samples/sample_repair_text.csv
```

全量本地数据默认放置在 `data/raw/`，该目录已加入 `.gitignore`，不会随仓库上传。标准化后的数据字段如下：

| 字段 | 含义 |
|------|------|
| `text` | 设备报修文本 |
| `fault_category` | 故障大类 |
| `risk_level` | 停机风险等级 |
| `department` | 推荐处理部门 |

标签体系保存在 [configs/labels.json](configs/labels.json)，包括 10 个故障大类、4 个风险等级和 10 个处理部门。训练切分时使用三任务组合标签进行分层，尽量保持 `train / val / test` 中的联合分布一致。

---

## 0x03. 模型与技术路线

项目保留两条建模路径：

| 路线 | 作用 | 说明 |
|------|------|------|
| 轻量 baseline | 快速验证数据与工程闭环 | 基于字符 n-gram 的多任务朴素贝叶斯，不依赖 GPU |
| BERT 多任务模型 | 面向正式文本语义建模 | 共享中文 BERT 编码器，后接三个分类头；当前保留训练入口 |

BERT 方案的核心结构为：

```text
中文报修文本
    |
Tokenizer
    |
BERT Encoder
    |
共享文本表示
    |
fault_category head / risk_level head / department head
```

训练阶段关注的不只是 accuracy，还包括 macro-F1、P0/P1 高风险召回、三任务全对率和部门混淆情况。对于实际业务场景，P0/P1 召回优先级高于单纯的整体准确率。

---

## 0x04. 技术架构与核心流程

```text
原始报修文本数据
    |
    v
Step 1: 数据标准化
    - 将四列原始 TXT/TSV 转换为带表头 CSV
    - 校验字段数量、空字段和编码
    |
    v
Step 2: 数据探索分析
    - 统计标签分布、文本长度、组合标签分布
    - 检查重复文本、冲突标签和疑似泄漏短语
    |
    v
Step 3: 清洗与分层切分
    - 删除空字段、完全重复行和同文本多标签冲突样本
    - 生成 label schema
    - 按组合标签切分 train / val / test
    |
    v
Step 4: 多任务模型训练
    - baseline: 字符 n-gram Naive Bayes
    - optional: BERT shared encoder + three classification heads
    |
    v
Step 5: 评估与推理
    - 输出三任务指标、P0/P1 召回和混淆统计
    - 支持单条报修文本命令行预测
```

---

## 0x05. 运行方式

安装本地包：

```powershell
cd E:\Github\industrial-fault-text-classifier
python -m pip install -e .
```

快速跑通公开样例闭环：

```powershell
industrial-fault-go
```

按步骤运行：

```powershell
python scripts\step1_convert_to_csv.py --input data\raw\manufacturing_repair_text_dataset_cn.txt --output data\processed\standard_dataset.csv
python scripts\step2_dataset_eda.py --input data\processed\standard_dataset.csv --report artifacts\reports\eda_report.json
python scripts\step3_clean_and_split.py --input data\processed\standard_dataset.csv --output-dir data\processed\splits --labels data\processed\labels.json --report artifacts\reports\split_report.json
python scripts\step4_model_training.py --train data\processed\splits\train.csv --val data\processed\splits\val.csv --labels data\processed\labels.json --model-dir artifacts\models\baseline --backend naive_bayes --max-train-samples 2000
python scripts\step5_evaluate_and_predict.py evaluate --model-dir artifacts\models\baseline --data data\processed\splits\test.csv --report artifacts\reports\eval_report.json --predictions artifacts\reports\predictions.csv
python scripts\step5_evaluate_and_predict.py predict --model-dir artifacts\models\baseline --text "空压机运行中压力波动明显，主线节拍受到影响，请安排检修。"
```

当前公开的 `industrial-fault-go`、Step 5 评估和单条预测闭环使用 `naive_bayes` 后端，便于在无 GPU 和无本地预训练模型缓存的环境中快速验证流程。BERT 后端已保留 Step 4 训练入口；统一评估与推理接口需在模型产物格式稳定后继续接入。

---

## 0x06. 项目文件结构

```text
industrial-fault-text-classifier/
├── configs/
│   ├── labels.json
│   └── train_config.json
├── data/
│   ├── README.md
│   ├── raw/                         # 本地全量数据，默认不上传
│   └── samples/
│       └── sample_repair_text.csv    # 公开样例数据
├── docs/
│   ├── step1_dataset_prepare.md
│   ├── step2_dataset_eda.md
│   ├── step3_cleaning_stratified_split.md
│   ├── step4_model_training.md
│   └── step5_evaluation_inference.md
├── scripts/
│   ├── step1_convert_to_csv.py
│   ├── step2_dataset_eda.py
│   ├── step3_clean_and_split.py
│   ├── step4_model_training.py
│   └── step5_evaluate_and_predict.py
├── src/
│   └── industrial_fault_classifier/
│       ├── baseline.py
│       ├── bert.py
│       ├── cli.py
│       ├── data.py
│       ├── evaluation.py
│       ├── inference.py
│       ├── labels.py
│       ├── metrics.py
│       ├── pipeline.py
│       └── training.py
├── README.md
├── README_EN.md
└── pyproject.toml
```

---

## 0x07. 项目价值与后续方向

本项目的核心价值在于把设备报修文本转化为结构化标签，使维修派单、风险分级和故障统计具备自动化基础。相比单任务分类，多任务结构能够共享文本语义，并同时建模“故障现象、停机风险、处理部门”之间的业务关联。

后续可继续推进以下方向：

1. 引入经过授权和脱敏处理的真实企业工单样本。
2. 建立更严格的风险等级标注规范，重点提升 P0/P1 召回。
3. 增加混淆矩阵分析，定位处理部门之间的误分边界。
4. 使用 MacBERT、RoBERTa-wwm-ext 等中文预训练模型进行对比。
5. 建立低置信度人工复核机制，将修正样本回流到训练集。
