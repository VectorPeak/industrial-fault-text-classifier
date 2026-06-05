<div align="center">

# Industrial Fault Text Classifier | 化工业设备报修文本多任务分类系统

面向化工业报修文本的故障大类、停机风险等级与处理部门多任务分类方案

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

> 本仓库公开数据基于 Kaggle 公开企业数据集进行清洗与增强构建，仅用于复现项目流程、验证建模方案与展示工程实现；数据中不包含企业真实生产工单、设备编号、人员信息或现场敏感信息。如项目在实际企业场景中落地，真实报修记录及中间数据产物应按企业数据安全要求处理，不在公开仓库中发布。

## 0x01. 项目背景

化工生产现场的报修记录、点检记录和维修工单通常以自由文本形式记录。一条工单文本中可能同时包含设备对象、部件位置、故障现象、异常程度、停机影响和建议处理方向，例如“压缩机轴承温度持续升高，伴随振动增大，需尽快安排机修检查”。传统处理方式主要依赖调度人员逐条阅读文本后进行人工识别与分派，容易受到文本表述不规范、人员经验差异、值守时段压力以及高风险工单集中出现等因素影响，进而造成故障类别判断不一致、响应优先级滞后或处理部门分派不准确。

本项目将单条报修文本转换为三个结构化判断结果：

| 任务 | 输出 | 用途 |
|------|------|------|
| 故障大类分类 | 机械故障、电气故障、传感器故障等 | 形成维修统计与故障画像 |
| 停机风险等级识别 | P0 / P1 / P2 / P3 | 辅助识别高优先级工单 |
| 处理部门推荐 | 机械维修、电气维修、自动化工程师等 | 降低转派和等待成本 |

项目定位不是替代工程师判断，而是将非结构化报修文本转化为可检索、可统计、可辅助派单的结构化信息。

---

## 0x02. 数据集与标签体系

当前公开仓库提交基于 Kaggle 公开企业数据集清洗增强后的全量 CSV，并保留小规模样例用于快速 smoke test：

```text
data/full/chemical_repair_text_dataset_cn.csv
data/samples/sample_repair_text.csv
```

全量 CSV 约 22 万行，采用带表头的标准 CSV 格式。`data/raw/` 仅用于存放本地原始 TXT/TSV 来源文件，该目录已加入 `.gitignore`。标准化后的数据字段如下：

| 字段 | 含义 |
|------|------|
| `text` | 设备报修文本 |
| `fault_category` | 故障大类 |
| `risk_level` | 停机风险等级 |
| `department` | 推荐处理部门 |

标签体系保存在 [configs/labels.json](configs/labels.json)，包括 10 个故障大类、4 个风险等级和 10 个处理部门。训练切分时使用三任务组合标签进行分层，尽量保持 `train / val / test` 中的联合分布一致。

### 2.1 EDA

为了确认多任务标签分布、文本长度分布以及标签之间的组合关系，项目将 Step 2 的主要 EDA 结果整理为一张组合图。图中包含故障大类分布、停机风险等级分布、推荐处理部门分布、文本长度分布，以及 `fault_category × risk_level`、`department × fault_category` 两组交叉热力图。

![数据集 EDA Dashboard](artifacts/figures/eda_dashboard.png)

### 2.2 数据质量控制与清洗

在模型训练前，项目将报修文本、标签字段和数据切分结果纳入统一的数据质量控制流程。Step 3 不仅执行基础清洗操作，也对监督信号一致性、潜在标签泄漏和三任务联合分布进行审计，以降低噪声样本对分类边界和评估结论的影响。

| 质量项 | 风险说明 | 控制策略 |
|------|------|------|
| 关键字段缺失 | `text`、`fault_category`、`risk_level` 或 `department` 任一字段缺失时，样本无法形成完整的多任务监督信号 | 剔除关键字段为空的记录，并保留清洗前后的样本计数 |
| 完全重复记录 | 文本与三组标签完全一致的重复样本会放大局部模式权重，影响训练集分布 | 保留首条有效记录，删除重复副本，避免重复样本影响模型拟合 |
| 同文本标签冲突 | 相同报修文本对应多组不同标签，说明标注口径存在歧义或监督信号不一致 | 剔除冲突文本对应的全部样本，保证训练标签的可学习性与评估口径稳定 |
| 文本格式噪声 | 换行、制表符和连续空格会增加分词及字符 n-gram 特征的稀疏性 | 统一折叠空白字符并去除首尾空白，在不改变语义的前提下规范输入文本 |
| 潜在标签泄漏 | 文本中直接出现 `P0/P1`、处理部门名称等显式标签标识，可能导致模型学习非业务语义捷径 | 在 EDA 阶段统计疑似泄漏短语，作为数据源质量和后续脱敏规则的审计依据 |
| 切分分布偏移 | 仅按单一任务标签切分可能破坏 `fault_category`、`risk_level` 和 `department` 的联合分布，使验证指标失真 | 使用三任务组合标签进行分层切分，并保留各子集标签分布对比结果 |

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
原始报修文本数据 (4列: text / fault_category / risk_level / department)
    |
    v
Step 1: 数据标准化与CSV固化
    ├── 读取原始TXT/TSV四列数据，统一字段名与编码
    ├── 校验字段数量、空字段、异常行和CSV转义
    ├── 输出全量CSV: data/full/chemical_repair_text_dataset_cn.csv (220,000行)
    └── 输出样例CSV: data/samples/sample_repair_text.csv (200行, 用于smoke test)
    |
    v
Step 2: 数据探索分析与质量审计
    ├── 统计三任务标签分布: 故障大类(10类) / 风险等级(4类) / 处理部门(10类)
    ├── 统计文本长度分布: min / max / mean / p50 / p90 / p95
    ├── 统计组合标签分布: fault_category + risk_level + department
    ├── 检查重复文本、同文本多标签冲突和疑似标签泄漏短语
    └── 输出报告: artifacts/reports/eda_report.json
    |
    v
Step 3: 数据清洗与分层切分
    ├── 删除空字段样本、完全重复行和同文本多标签冲突样本
    ├── 清洗结果: 220,000行 -> 219,160行
    ├── 生成标签映射: data/processed/labels.json
    ├── 按组合标签进行分层切分，保持三任务联合分布稳定
    └── 输出切分: train(175,272) / val(21,852) / test(22,036)
    |
    v
Step 4: 多任务文本分类模型训练
    ├── baseline: 字符n-gram特征 + 多任务Naive Bayes
    ├── BERT: shared encoder + fault/risk/department三个分类头
    ├── 微调策略: BERT encoder与分类头共同训练(全量微调)
    ├── 训练配置: max_length / batch_size / learning_rate / loss_weights
    └── 输出模型: artifacts/models/{backend}/model.pkl 或 model.pt
    |
    v
Step 5: 多任务评估与命令行推理
    ├── 评估指标: accuracy / macro-F1 / three-task exact match
    ├── 风险指标: P0/P1高风险召回
    ├── 错误分析: expected-predicted混淆组合统计
    ├── 输出报告: artifacts/reports/eval_report.json + predictions.csv
    └── 单条推理: 输入报修文本 -> 输出故障大类 / 风险等级 / 处理部门
```

---

## 0x05. 最终价值与落地场景

### 5.1 离线评估口径

本项目的评估重点不是单一准确率，而是围绕化工报修场景中的“故障识别、风险优先级、派单准确性”建立多任务评估口径。公开仓库中的 `industrial-fault-go` 默认使用小规模样例数据进行 smoke test，主要用于验证工程闭环；正式性能应在经过授权、脱敏并完成标注一致性审计的企业工单数据上重新评估。

| 评估对象 | 核心指标 | 价值解读 |
|------|------|------|
| 故障大类分类 | accuracy、macro-F1、类别混淆统计 | 判断模型能否稳定识别机械、电气、仪表、工艺、安全等主要故障类型，为故障统计和设备画像提供结构化输入 |
| 停机风险等级 | P0/P1 召回、macro-F1、风险等级混淆统计 | 高风险工单的漏判成本高于普通误分，因此 P0/P1 召回优先级高于整体准确率 |
| 处理部门推荐 | department accuracy、部门混淆矩阵 | 评估模型对维修、自动化、工艺、安全等处理部门的分派能力，减少跨部门转派和等待时间 |
| 三任务联合结果 | three-task exact match、组合标签分布稳定性 | 衡量“故障类型 + 风险等级 + 处理部门”是否能够同时预测正确，更接近真实派单场景的端到端效果 |

### 5.2 业务价值

从“人工逐条判读”到“结构化预分派”：

1. 将非结构化报修文本转换为标准标签，降低工单初筛对个人经验和班次状态的依赖。
2. 对 P0/P1 高风险工单进行前置识别，使调度人员可以优先处理可能影响安全、停机或产线连续性的事件。
3. 将历史报修记录沉淀为可统计的数据资产，支持故障类型分析、部门工作量分析和设备薄弱环节识别。
4. 在工单创建阶段给出推荐处理部门，减少反复转派、沟通确认和响应延迟。

### 5.3 系统落地场景

工单系统辅助派单：

接入 EAM、CMMS 或企业自建维修工单系统，在工单提交后自动返回故障大类、风险等级和推荐处理部门。低置信度样本进入人工复核，高置信度样本可作为派单建议展示给调度人员。

高风险报修队列：

将 P0/P1 预测结果写入工单优先级字段，形成高风险报修队列。调度人员可以按风险等级、故障类型和责任部门进行筛选，避免关键工单被普通报修淹没。

故障知识沉淀：

持续统计故障大类、设备对象、处理部门和风险等级之间的关系，形成面向设备管理的文本化故障画像。后续可进一步扩展到关键词抽取、设备实体识别和维修知识库构建。

人机协同标注闭环：

对模型低置信度、三任务预测不一致或高风险误分样本进行人工复核，并将修正结果回流训练集。该闭环可以逐步提升标签规范性和模型对企业内部表达习惯的适应能力。

### 5.4 后续演进方向

1. 引入经过授权和脱敏处理的真实企业工单样本，建立更贴近生产环境的验证集。
2. 完善风险等级标注规范，重点提升 P0/P1 高风险工单召回。
3. 增加混淆矩阵、置信度分布和低置信度样本分析，定位故障类型和处理部门之间的误分边界。
4. 对比 MacBERT、RoBERTa-wwm-ext 等中文预训练模型，并评估全量微调、冻结编码器和轻量化蒸馏方案。
5. 建立低置信度人工复核与主动学习机制，使修正样本能够持续回流到训练数据中。

---

## 0x06. 运行方式

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
python scripts\step2_dataset_eda.py --input data\full\chemical_repair_text_dataset_cn.csv --report artifacts\reports\eda_report.json
python scripts\step3_clean_and_split.py --input data\full\chemical_repair_text_dataset_cn.csv --output-dir data\processed\splits --labels data\processed\labels.json --report artifacts\reports\split_report.json
python scripts\step4_model_training.py --train data\processed\splits\train.csv --val data\processed\splits\val.csv --labels data\processed\labels.json --model-dir artifacts\models\baseline --backend naive_bayes --max-train-samples 2000
python scripts\step5_evaluate_and_predict.py evaluate --model-dir artifacts\models\baseline --data data\processed\splits\test.csv --report artifacts\reports\eval_report.json --predictions artifacts\reports\predictions.csv
python scripts\step5_evaluate_and_predict.py predict --model-dir artifacts\models\baseline --text "空压机运行中压力波动明显，主线节拍受到影响，请安排检修。"
```

如需从本地原始 TXT/TSV 重新生成全量 CSV，可运行：

```powershell
python scripts\step1_convert_to_csv.py --input data\raw\chemical_repair_text_dataset_cn.txt --output data\full\chemical_repair_text_dataset_cn.csv
```

当前公开的 `industrial-fault-go`、Step 5 评估和单条预测闭环使用 `naive_bayes` 后端，便于在无 GPU 和无本地预训练模型缓存的环境中快速验证流程。BERT 后端已保留 Step 4 训练入口；统一评估与推理接口需在模型产物格式稳定后继续接入。

---

## 0x07. 项目文件结构

```text
industrial-fault-text-classifier/
├── configs/                                      # 配置文件与标签体系
│   ├── labels.json                               # 三个任务的 label2id / id2label 映射
│   └── train_config.json                         # 数据路径、切分比例、baseline 与 BERT 训练参数
│
├── data/                                         # 数据目录
│   ├── README.md                                 # 数据目录说明
│   ├── full/
│   │   └── chemical_repair_text_dataset_cn.csv   # Kaggle公开企业数据集清洗增强后的全量 CSV，约 22 万行
│   ├── raw/                                      # 本地原始 TXT/TSV 来源文件，默认不上传
│   └── samples/
│       └── sample_repair_text.csv                # 小规模公开样例，用于快速 smoke test
│
├── docs/                                         # 分步骤技术说明
│   ├── step1_dataset_prepare.md                  # Step 1：数据标准化与 CSV 生成
│   ├── step2_dataset_eda.md                      # Step 2：标签分布、文本长度与泄漏风险分析
│   ├── step3_cleaning_stratified_split.md        # Step 3：清洗、去重、冲突剔除与分层切分
│   ├── step4_model_training.md                   # Step 4：baseline 与 BERT 多任务训练说明
│   └── step5_evaluation_inference.md             # Step 5：评估指标与单条文本推理
│
├── scripts/                                      # 可直接运行的步骤脚本
│   ├── step1_convert_to_csv.py                   # 调用 CLI convert，将原始数据转为标准 CSV
│   ├── step2_dataset_eda.py                      # 调用 CLI eda，生成数据分析报告
│   ├── step3_clean_and_split.py                  # 调用 CLI split，生成 train / val / test
│   ├── step4_model_training.py                   # 调用 CLI train，训练 baseline 或 BERT
│   └── step5_evaluate_and_predict.py             # 调用 CLI evaluate / predict，评估或推理
│
├── src/
│   └── industrial_fault_classifier/              # 核心 Python 包
│       ├── __init__.py                           # 包版本与模块声明
│       ├── baseline.py                           # 字符 n-gram 多任务 Naive Bayes baseline
│       ├── bert.py                               # BERT 共享编码器 + 三分类头训练入口
│       ├── cli.py                                # 命令行参数解析与子命令分发
│       ├── config.py                             # 项目路径、JSON 配置读写工具
│       ├── constants.py                          # 数据列名、任务名、默认预测文本
│       ├── data.py                               # CSV/TSV 读写、校验、清洗与分层切分
│       ├── evaluation.py                         # 模型加载、测试集评估与预测结果导出
│       ├── inference.py                          # 单条报修文本推理封装
│       ├── labels.py                             # 标签映射生成、保存、加载与解码
│       ├── metrics.py                            # accuracy、macro-F1、P0/P1 召回等指标
│       ├── pipeline.py                           # Step 1-5 端到端流程编排
│       └── training.py                           # baseline / BERT 训练统一入口
│
├── artifacts/                                    # 本地运行产物目录
│   └── README.md                                 # 说明模型、报告、图表等生成产物的存放规则
├── .gitignore                                    # 忽略 raw、processed、模型、报告、缓存等文件
├── README.md                                     # 中文项目说明
├── README_EN.md                                  # 英文项目说明
└── pyproject.toml                                # Python 包配置、依赖与命令行入口
```

---

## 0x08. 版本说明

当前版本重点完成项目结构整理、公开 CSV 数据固化、分步骤脚本封装、baseline 训练闭环和 README 文档体系建设。BERT 多任务模型保留训练入口，后续可在模型产物格式、评估接口和推理服务接口稳定后接入完整评估流程。
