<div align="center">

# Industrial Fault Text Classifier | Chemical Equipment Repair Text Multi-Task Classification System

A multi-task classification pipeline for fault category, downtime risk level, and responsible department prediction from chemical repair text.

![python](https://img.shields.io/badge/python-3.10+-3776AB)
![pytorch](https://img.shields.io/badge/PyTorch-optional--BERT-EE4C2C)
![transformers](https://img.shields.io/badge/Transformers-BERT-blue)
![multitask](https://img.shields.io/badge/multi--task-learning-5C7CFA)
![maintenance](https://img.shields.io/badge/industrial-maintenance-green)

[简体中文](README.md) | English

</div>

```text
repair text -> label mapping -> data quality check -> stratified split -> multi-task classifier -> evaluation -> inference demo
```

> The public dataset in this repository is built by cleaning and augmenting a public Kaggle enterprise dataset. It is used only to reproduce the project workflow, validate the modeling approach, and demonstrate the engineering implementation. It does not contain real enterprise production work orders, equipment identifiers, personnel information, or sensitive site records. If this project is deployed in an enterprise environment, real repair records and intermediate data artifacts should be handled according to enterprise data security requirements and should not be published in a public repository.

## 0x01. Project Background

Repair records, inspection notes, and maintenance work orders in chemical production sites are usually recorded as text. A single work order may contain equipment objects, component locations, fault symptoms, abnormal severity, downtime impact, and suggested handling direction. For example: "The compressor bearing temperature keeps rising, accompanied by increasing vibration; mechanical inspection should be arranged as soon as possible." Traditional workflows rely mainly on dispatchers reading each text record and manually assigning categories and departments. This process can be affected by inconsistent wording, differences in personnel experience, workload during duty shifts, and the accumulation of high-risk tickets, which may lead to inconsistent fault classification, delayed priority judgment, or inaccurate department routing.

This project converts one repair text into three structured prediction results:

| Task | Output | Purpose |
|------|--------|---------|
| Fault category classification | Mechanical fault, electrical fault, sensor fault, etc. | Build maintenance statistics and fault profiles |
| Downtime risk identification | P0 / P1 / P2 / P3 | Assist high-priority work order identification |
| Responsible department recommendation | Mechanical maintenance, electrical maintenance, automation engineer, etc. | Reduce reassignment and waiting cost |

The project is not intended to replace engineering judgment. Its purpose is to convert unstructured repair text into structured information that can be searched, counted, and used to support dispatch decisions.

---

## 0x02. Dataset And Label Schema

The public repository currently includes a full CSV dataset cleaned and augmented from a public Kaggle enterprise dataset, together with a small sample dataset for quick smoke tests:

```text
data/full/chemical_repair_text_dataset_cn.csv
data/samples/sample_repair_text.csv
```

The full CSV contains about 220,000 rows and uses a standard headered CSV format. `data/raw/` is reserved for local raw TXT/TSV source files and is ignored by `.gitignore`. The standardized dataset fields are:

| Field | Meaning |
|-------|---------|
| `text` | Equipment repair text |
| `fault_category` | Fault category |
| `risk_level` | Downtime risk level |
| `department` | Recommended responsible department |

The label schema is stored in [configs/labels.json](configs/labels.json), covering 10 fault categories, 4 risk levels, and 10 responsible departments. Dataset splitting uses the combined three-task label for stratification, so the joint distribution in `train / val / test` remains as stable as possible.

### 2.1 EDA

To inspect multi-task label distributions, text length distribution, and label relationships, the main Step 2 EDA results are summarized in one dashboard. The figure includes fault category distribution, downtime risk level distribution, responsible department distribution, text length distribution, and two cross heatmaps: `fault_category x risk_level` and `department x fault_category`.

![Dataset EDA Dashboard](https://github.com/VectorPeak/industrial-fault-text-classifier/blob/main/artifacts/figures/eda_dashboard.png?raw=true)

### 2.2 Data Quality Control And Cleaning

Before model training, the project brings repair text, label fields, and dataset split results into a unified data quality control process. Step 3 not only performs basic cleaning, but also audits supervision consistency, potential label leakage, and the joint distribution of the three tasks. This reduces the impact of noisy samples on classification boundaries and evaluation conclusions.

| Quality Item | Risk | Control Strategy |
|--------------|------|------------------|
| Missing key fields | If any of `text`, `fault_category`, `risk_level`, or `department` is missing, the sample cannot provide complete multi-task supervision | Remove records with empty key fields and keep before/after sample counts |
| Exact duplicate records | Records with identical text and identical three labels may amplify local patterns and distort training distribution | Keep the first valid record and remove duplicate copies |
| Same-text label conflicts | The same repair text mapped to different label triples indicates ambiguous labeling or inconsistent supervision | Remove all samples corresponding to conflicting texts to keep labels learnable and evaluation stable |
| Text formatting noise | Line breaks, tabs, and repeated spaces increase tokenization and character n-gram sparsity | Normalize whitespace without changing semantic content |
| Potential label leakage | Direct labels such as `P0/P1` or responsible department names in the text may cause the model to learn shortcuts rather than business semantics | Count suspected leakage phrases during EDA as an audit signal for data source quality and later desensitization rules |
| Split distribution shift | Splitting by only one task label may break the joint distribution of `fault_category`, `risk_level`, and `department`, making validation metrics misleading | Use the combined three-task label for stratified splitting and retain split distribution comparison results |

---

## 0x03. Model And Technical Selection Process

### 3.1 Stage 1: Applicable Boundary Of Rule-Based Dispatch

At an early stage, the project considered using keyword dictionaries, fault phrase libraries, and department mapping rules to classify repair text, judge risk, and recommend handling departments. For example, when terms such as "bearing", "vibration", and "temperature rise" appear in the text, the record could be assigned to mechanical faults; when expressions such as "interlock", "DCS", and "valve position feedback" appear, it could be assigned to control-system or instrumentation-related categories.

This approach is simple to implement, interpretable, and low-cost to deploy, so it is useful as an initial data audit tool. However, in real chemical production text environments, rule-based methods have clear limitations:

1. Site records contain many abbreviations, informal descriptions, synonyms, and incomplete expressions, so rule coverage costs continue to grow with data scale.
2. P0/P1 high-risk levels often depend on the combined judgment of equipment object, fault symptom, impact scope, and duration, rather than a single keyword trigger.
3. Fault categories and responsible departments are not strictly one-to-one. The same type of fault may be routed to different teams depending on equipment location, risk level, and production impact.
4. Rule systems are easy to explain and audit, but generalization is limited, making long-term iteration and cross-scenario migration difficult.

Therefore, this project positions rule-based methods as tools for data audit, label leakage inspection, and manual review assistance, rather than the main modeling route.

### 3.2 Stage 2: Model Route Comparison

Before finalizing the solution, the project compared four technical routes:

| Route | Representative Methods | Strengths | Limitations | Role In This Project |
|-------|------------------------|-----------|-------------|----------------------|
| Keyword rules | Dictionary matching, regex rules, manual mapping tables | Strong interpretability, low deployment cost, easy audit | High maintenance cost, difficult to cover synonyms, implicit risk, and complex context | Used for data audit and leakage checks, not as the main model |
| Traditional machine learning | TF-IDF / character n-gram + Naive Bayes / Linear SVM | Fast training, few dependencies, suitable for quickly validating the engineering loop | Limited ability to model long-range semantics, contextual combinations, and complex expressions | Default public-repository baseline |
| Pretrained language models | Chinese BERT / MacBERT / RoBERTa-wwm-ext | Can model contextual semantics and handle complex repair expressions | Higher training cost, requires GPU, model cache, and stricter tuning | Formal semantic modeling route |
| Large language model classification | Prompt classification, few-shot classification, external inference APIs | Fast cold start and can produce explanatory output | Cost, stability, data security, and batch inference controllability require separate evaluation | Suitable for later quality inspection, sampling, or assisted review |

The final design uses a dual-route structure: "lightweight baseline + BERT multi-task model entry." The baseline validates the data, scripts, evaluation, and inference loop quickly; the BERT route is used for formal semantic modeling when compute resources and authorized data are available.

### 3.3 Stage 3: From Single-Task Classification To Multi-Task Joint Modeling

An initial feasible approach was to train three independent classifiers: one for fault category, one for downtime risk level, and one for responsible department. This is easy to implement, but it separates business relationships among the three tasks.

This project ultimately maps one repair text into three structured labels at the same time:

```text
Repair text
    |
Shared text features / shared semantic representation
    |
fault_category / risk_level / department
```

The reasons for using a multi-task design include:

1. Fault category and responsible department have strong business relationships. For example, mechanical faults are more likely to enter the mechanical maintenance queue, while control-system faults are more likely to enter automation or instrumentation queues.
2. Downtime risk level affects dispatch priority, review strategy, and processing time limits.
3. Joint three-task output is closer to the real dispatch process and is more useful than predicting only one label.
4. The BERT route can reuse text semantics through a shared encoder, while three classification heads learn task-specific boundaries.

The current public baseline uses the same character n-gram feature logic and trains lightweight classifiers separately for the three tasks. The BERT entry uses a shared Chinese BERT encoder and three classification heads, with joint fine-tuning of the encoder and heads.

### 3.4 Stage 4: Dataset Split Strategy Adjustment

If the dataset is stratified by only one label, such as `fault_category`, the joint distribution of `risk_level` and `department` in the training, validation, and test sets may shift. For a multi-task dispatch system, this reduces how well validation metrics represent the real usage scenario.

Therefore, Step 3 uses the combined three-task label for stratified splitting:

```python
stratify_key = fault_category + risk_level + department
```

The goal is not to pursue superficial balance for each single-task label, but to preserve real dispatch combinations in `train / val / test` as much as possible, making evaluation results closer to the end-to-end usage scenario.

### 3.5 Stage 5: From Accuracy To Risk-Oriented Evaluation

The initial evaluation design can easily focus only on overall accuracy. In chemical repair scenarios, however, different errors have different business costs: missing a P0/P1 high-risk work order is more serious than misclassifying an ordinary P2/P3 ticket; a wrong responsible department increases reassignment and waiting costs; and an error in any of the three tasks may affect the final dispatch decision.

Therefore, the evaluation scope is expanded to:

1. `accuracy`: observes each task's overall classification ability.
2. `macro-F1`: prevents high-frequency classes from masking low-frequency class performance.
3. `P0/P1 recall`: focuses on whether high-risk work orders are identified in time.
4. `three-task exact match`: measures whether all three predictions are correct at the same time.
5. Confusion combination statistics: locates typical misclassification boundaries among fault category, risk level, and responsible department.

This project does not use a single highest accuracy score as the only goal. Instead, it builds an evaluation framework around dispatch risk, high-risk recall, review cost, and department reassignment cost.

### 3.6 Stage 6: From Offline Classification To Production Decision Support

An offline model can only complete the basic conversion from text to labels. For enterprise deployment, a more reasonable approach is to use model output as dispatch suggestions and risk prompts, rather than directly replacing engineering judgment.

The recommended production usage flow is:

```text
New repair text
    |
Model outputs three task labels
    |
Confidence and risk-level judgment
    |
High-risk or low-confidence samples enter review queue
    |
Review results flow back into training data
```

This design preserves the necessary human review authority in safety-critical production scenarios, while allowing the model to handle high-frequency, repetitive, and standardized initial screening tasks. If real enterprise work orders are later integrated, priority should be given to low-confidence sample feedback, P0/P1 misclassification audit, department misrouting analysis, and periodic retraining mechanisms.

---

## 0x04. Technical Architecture And Core Pipeline

```text
Raw repair text data (4 columns: text / fault_category / risk_level / department)
    |
    v
Step 1: Data standardization and CSV solidification
    |-- Read raw TXT/TSV four-column data and unify field names and encoding
    |-- Validate column count, empty fields, abnormal rows, and CSV escaping
    |-- Output full CSV: data/full/chemical_repair_text_dataset_cn.csv (220,000 rows)
    `-- Output sample CSV: data/samples/sample_repair_text.csv (200 rows, for smoke tests)
    |
    v
Step 2: Dataset exploration and quality audit
    |-- Count three-task label distributions: fault categories (10) / risk levels (4) / departments (10)
    |-- Summarize text length distribution: min / max / mean / p50 / p90 / p95
    |-- Count combined label distribution: fault_category + risk_level + department
    |-- Check duplicate texts, same-text multi-label conflicts, and suspected label leakage phrases
    `-- Output report: artifacts/reports/eda_report.json
    |
    v
Step 3: Data cleaning and stratified splitting
    |-- Remove empty-field samples, exact duplicates, and same-text multi-label conflicts
    |-- Cleaning result: 220,000 rows -> 219,160 rows
    |-- Generate label mapping: data/processed/labels.json
    |-- Split by combined labels to keep the three-task joint distribution stable
    `-- Output splits: train(175,272) / val(21,852) / test(22,036)
    |
    v
Step 4: Multi-task text classification model training
    |-- baseline: character n-gram features + multi-task Naive Bayes
    |-- BERT: shared encoder + fault/risk/department heads
    |-- Fine-tuning strategy: train BERT encoder and heads together (full fine-tuning)
    |-- Training config: max_length / batch_size / learning_rate / loss_weights
    `-- Output model: artifacts/models/{backend}/model.pkl or model.pt
    |
    v
Step 5: Multi-task evaluation and command-line inference
    |-- Evaluation metrics: accuracy / macro-F1 / three-task exact match
    |-- Risk metric: P0/P1 high-risk recall
    |-- Error analysis: expected-predicted confusion combination statistics
    |-- Output reports: artifacts/reports/eval_report.json + predictions.csv
    `-- Single-text inference: input repair text -> fault category / risk level / department
```

The corresponding step-by-step technical documents are:

| Document | Topic | Key Content |
|----------|-------|-------------|
| [step1_dataset_prepare.md](docs/step1_dataset_prepare.md) | Data standardization and CSV solidification | Raw TXT/TSV/CSV reading, four-field unification, sample generation |
| [step2_dataset_eda.md](docs/step2_dataset_eda.md) | EDA and quality audit | Label distribution, text length, combined labels, duplicate texts, leakage phrase checks |
| [step3_cleaning_stratified_split.md](docs/step3_cleaning_stratified_split.md) | Data cleaning and stratified splitting | Empty fields, duplicate samples, conflicting labels, and three-task combined stratification |
| [step4_model_training.md](docs/step4_model_training.md) | Multi-task model training | Character n-gram baseline, BERT shared encoder, and three classification heads |
| [step5_evaluation_inference.md](docs/step5_evaluation_inference.md) | Evaluation and inference | accuracy, macro-F1, three-task exact match, confusion combinations, and single-text prediction |

---

## 0x05. Final Value And Deployment Scenarios

### 5.1 Prediction Performance

This project maps one repair text into three structured results: fault category, downtime risk level, and responsible department. The following metrics come from the current multi-task BERT training result and describe the model's offline validation performance.

| Prediction Target | Metric | Result | Value Interpretation |
|-------------------|--------|-------:|----------------------|
| Fault category | accuracy / macro-F1 | 92.40% / 92.42% | Stably distinguishes major fault types such as mechanical, electrical, instrumentation, process, and safety faults |
| Downtime risk level | accuracy / macro-F1 | 85.84% / 86.09% | Helps identify P0/P1 high-risk work orders and provides model evidence for priority ranking |
| Responsible department | accuracy / macro-F1 | 90.23% / 90.19% | Provides candidate recommendations for maintenance, automation, process, safety, and other departments |
| Three-task overall | Average macro-F1 | 89.57% | Supports the joint prediction flow of fault identification, risk judgment, and department recommendation |

### 5.2 Business Value

1. Convert natural-language repair records into structured labels, reducing the dependence of initial work order screening on individual experience and text expression differences.
2. Identify high-risk work orders early, allowing dispatchers to prioritize events that may affect safety, downtime, or production-line continuity.
3. Provide responsible department recommendations when a work order is created, reducing reassignment, communication confirmation, and response delay costs.
4. Turn historical repair text into statistical data assets that support fault type analysis, department workload analysis, and weak-equipment identification.
5. Review low-confidence samples and high-risk misclassified samples manually, then feed corrected results back into the training set to continuously improve the model's adaptation to enterprise site language.

---

## 0x06. Usage

Install the local package:

```powershell
cd E:\Github\industrial-fault-text-classifier
python -m pip install -e .
```

Run the public-sample smoke pipeline:

```powershell
industrial-fault-go
```

Run the pipeline step by step:

```powershell
python scripts\step2_dataset_eda.py --input data\full\chemical_repair_text_dataset_cn.csv --report artifacts\reports\eda_report.json
python scripts\step3_clean_and_split.py --input data\full\chemical_repair_text_dataset_cn.csv --output-dir data\processed\splits --labels data\processed\labels.json --report artifacts\reports\split_report.json
python scripts\step4_model_training.py --train data\processed\splits\train.csv --val data\processed\splits\val.csv --labels data\processed\labels.json --model-dir artifacts\models\baseline --backend naive_bayes --max-train-samples 2000
python scripts\step5_evaluate_and_predict.py evaluate --model-dir artifacts\models\baseline --data data\processed\splits\test.csv --report artifacts\reports\eval_report.json --predictions artifacts\reports\predictions.csv
python scripts\step5_evaluate_and_predict.py predict --model-dir artifacts\models\baseline --text "The air compressor shows obvious pressure fluctuation during operation, affecting the main production line rhythm. Please arrange maintenance inspection."
```

If you need to regenerate the full CSV from local raw TXT/TSV files:

```powershell
python scripts\step1_convert_to_csv.py --input data\raw\chemical_repair_text_dataset_cn.txt --output data\full\chemical_repair_text_dataset_cn.csv
```

The public `industrial-fault-go`, Step 5 evaluation, and single-text prediction loop currently use the `naive_bayes` backend, making the workflow easy to verify without GPU or local pretrained model cache. The BERT backend is kept as the Step 4 training entry; the unified evaluation and inference interfaces can be connected after the model artifact format becomes stable.

---

## 0x07. Project Structure

```text
industrial-fault-text-classifier/
|-- configs/                                      # Configuration files and label schema
|   |-- labels.json                               # label2id / id2label mappings for the three tasks
|   `-- train_config.json                         # Data paths, split ratios, baseline and BERT training parameters
|
|-- data/                                         # Data directory
|   |-- README.md                                 # Data directory description
|   |-- full/
|   |   `-- chemical_repair_text_dataset_cn.csv   # Full CSV cleaned and augmented from a public Kaggle enterprise dataset, about 220,000 rows
|   |-- raw/                                      # Local raw TXT/TSV source files, not uploaded by default
|   `-- samples/
|       `-- sample_repair_text.csv                # Small public sample for quick smoke tests
|
|-- docs/                                         # Step-by-step technical documentation
|   |-- step1_dataset_prepare.md                  # Step 1: data standardization and CSV generation
|   |-- step2_dataset_eda.md                      # Step 2: label distribution, text length, and leakage risk analysis
|   |-- step3_cleaning_stratified_split.md        # Step 3: cleaning, deduplication, conflict removal, and stratified splitting
|   |-- step4_model_training.md                   # Step 4: baseline and BERT multi-task training notes
|   `-- step5_evaluation_inference.md             # Step 5: evaluation metrics and single-text inference
|
|-- scripts/                                      # Directly runnable step scripts
|   |-- step1_convert_to_csv.py                   # Calls CLI convert to transform raw data into standard CSV
|   |-- step2_dataset_eda.py                      # Calls CLI eda to generate dataset analysis report
|   |-- step3_clean_and_split.py                  # Calls CLI split to generate train / val / test
|   |-- step4_model_training.py                   # Calls CLI train to train baseline or BERT
|   `-- step5_evaluate_and_predict.py             # Calls CLI evaluate / predict for evaluation or inference
|
|-- src/
|   `-- industrial_fault_classifier/              # Core Python package
|       |-- __init__.py                           # Package version and module declaration
|       |-- baseline.py                           # Character n-gram multi-task Naive Bayes baseline
|       |-- bert.py                               # BERT shared encoder + three-head training entry
|       |-- cli.py                                # CLI argument parsing and subcommand dispatch
|       |-- config.py                             # Project paths and JSON configuration helpers
|       |-- constants.py                          # Data column names, task names, default prediction text
|       |-- data.py                               # CSV/TSV IO, validation, cleaning, and stratified splitting
|       |-- evaluation.py                         # Model loading, test-set evaluation, and prediction export
|       |-- inference.py                          # Single repair text inference wrapper
|       |-- labels.py                             # Label mapping generation, saving, loading, and decoding
|       |-- metrics.py                            # accuracy, macro-F1, P0/P1 recall, and related metrics
|       |-- pipeline.py                           # Step 1-5 end-to-end pipeline orchestration
|       `-- training.py                           # Unified baseline / BERT training entry
|
|-- artifacts/                                    # Local runtime artifact directory
|   `-- README.md                                 # Storage rules for models, reports, figures, and related artifacts
|-- .gitignore                                    # Ignores raw, processed, models, reports, cache, and related files
|-- README.md                                     # Chinese project documentation
|-- README_EN.md                                  # English project documentation
`-- pyproject.toml                                # Python package configuration, dependencies, and CLI entry points
```

---

## 0x08. Key Lessons

Text structuring comes before model training: repair text must first be converted into fault category, downtime risk level, and responsible department before dispatching, statistics, and equipment profiling can have stable inputs.

Multi-task modeling is closer to real dispatching: fault type answers "what happened", risk level answers "how urgent it is", and responsible department answers "who should handle it"; the three should stay in the same business decision chain.

Data quality determines the upper bound of the supervision signal: empty fields, duplicate samples, same-text multi-label conflicts, and label leakage directly contaminate the training target, so cleaning, auditing, and combined-label stratified splitting must happen first.

High-risk recall and feedback loops must be retained: missing P0/P1 tickets is more costly than ordinary misclassification, so evaluation should inspect high-risk recall separately; low-confidence, misclassified, and reassigned samples should also enter review and flow back into the training set.
