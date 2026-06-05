<div align="center">

# Industrial Fault Text Classifier | Chemical Industry Repair Text Multi-Task Classification System

A multi-task classification pipeline for fault category, downtime risk level, and responsible department prediction from chemical-industry repair text.

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

> The public data in this repository is constructed data for reproducing the experimental workflow. It does not contain real enterprise work orders, equipment identifiers, personnel information, or sensitive site records. In an enterprise deployment, real repair records and intermediate data artifacts should follow internal data security requirements and should not be published in a public repository.

## 0x01. Project Background

Chemical-industry repair records, inspection notes, and maintenance work orders are often written as natural language text. A single record may contain equipment, components, symptoms, risk level, and routing information. Manual dispatch depends on people reading the text and assigning the work order, which can be affected by inconsistent wording, experience gaps, and high-risk tickets being buried in ordinary queues.

This project converts one repair text into three structured outputs:

| Task | Output | Purpose |
|------|--------|---------|
| Fault category classification | Mechanical, electrical, sensor, control-system faults, etc. | Fault statistics and equipment profiling |
| Downtime risk identification | P0 / P1 / P2 / P3 | High-priority ticket triage |
| Department recommendation | Mechanical maintenance, electrical maintenance, automation engineer, etc. | Lower dispatch and reassignment cost |

The system is positioned as a decision-support pipeline, not as a replacement for engineering judgment.

---

## 0x02. Dataset And Label Schema

The repository now commits the full constructed dataset CSV and keeps a small sample for smoke tests:

```text
data/full/chemical_repair_text_dataset_cn.csv
data/samples/sample_repair_text.csv
```

The full CSV has about 220,000 rows and uses a headered standard CSV schema. `data/raw/` is reserved for local raw TXT/TSV source files and is ignored by Git. The standardized dataset schema is:

| Field | Meaning |
|-------|---------|
| `text` | Repair text |
| `fault_category` | Fault category |
| `risk_level` | Downtime risk level |
| `department` | Recommended responsible department |

The label schema is stored in [configs/labels.json](configs/labels.json), covering 10 fault categories, 4 risk levels, and 10 departments. Dataset splitting uses the combined three-task label for stratification.

---

## 0x03. Model And Technical Route

The repository keeps two modeling routes:

| Route | Role | Notes |
|-------|------|-------|
| Lightweight baseline | Fast engineering smoke test | Character n-gram multi-task Naive Bayes, no GPU required |
| BERT multi-task model | Formal semantic modeling | Shared Chinese BERT encoder with three classification heads; training entry is retained |

BERT structure:

```text
Chinese repair text
    |
Tokenizer
    |
BERT Encoder
    |
Shared representation
    |
fault_category head / risk_level head / department head
```

Evaluation focuses on macro-F1, P0/P1 recall, three-task exact match, and department confusion, not only accuracy.

---

## 0x04. Architecture And Pipeline

```text
Raw repair text
    |
    v
Step 1: Standardize dataset
    - Convert raw four-column TXT/TSV into headered CSV
    - Validate column count, empty fields, and encoding
    |
    v
Step 2: Dataset EDA
    - Summarize label distribution, text length, and label combinations
    - Check duplicate text, conflicting labels, and possible leakage phrases
    |
    v
Step 3: Cleaning and stratified split
    - Remove empty rows, exact duplicates, and same-text label conflicts
    - Generate label schema
    - Split train / val / test by combined labels
    |
    v
Step 4: Multi-task model training
    - baseline: character n-gram Naive Bayes
    - optional: BERT shared encoder + three classification heads
    |
    v
Step 5: Evaluation and inference
    - Export task metrics, P0/P1 recall, and confusion summaries
    - Support single-text command-line prediction
```

---

## 0x05. Final Value And Deployment Scenarios

### 5.1 Offline Evaluation Scope

This project is evaluated as a multi-task industrial dispatch pipeline, not as a single accuracy benchmark. The default `industrial-fault-go` command uses a small public sample for smoke testing and engineering verification. Formal performance should be re-evaluated on authorized, anonymized enterprise work-order data after label consistency and data quality audits.

| Evaluation Target | Key Metrics | Value Interpretation |
|------|------|------|
| Fault category classification | accuracy, macro-F1, class-level confusion statistics | Measures whether the model can distinguish major chemical-industry fault categories such as mechanical, electrical, instrumentation, process, and safety-related issues |
| Downtime risk level | P0/P1 recall, macro-F1, risk-level confusion statistics | Prioritizes high-risk ticket recall because missing P0/P1 work orders is more costly than ordinary misclassification |
| Responsible department recommendation | department accuracy, department confusion matrix | Measures whether the model can reduce dispatch delay and cross-department rerouting |
| Three-task joint output | three-task exact match, combined-label distribution stability | Evaluates whether fault category, risk level, and department can be predicted correctly as one end-to-end dispatch decision |

### 5.2 Business Value

From manual ticket reading to structured pre-dispatch:

1. Convert unstructured repair text into standardized labels and reduce dependence on individual experience during initial triage.
2. Surface P0/P1 high-risk work orders early so dispatchers can prioritize events that may affect safety, downtime, or production continuity.
3. Turn historical repair records into analyzable data assets for fault statistics, department workload analysis, and equipment weakness profiling.
4. Provide a recommended responsible department when a ticket is created, reducing repeated rerouting and communication overhead.

### 5.3 Deployment Scenarios

Work-order assisted dispatch:

Integrate with EAM, CMMS, or enterprise maintenance-ticket systems. After a repair ticket is submitted, the model returns fault category, risk level, and recommended department. Low-confidence samples can be routed to human review, while high-confidence outputs can be shown as dispatch suggestions.

High-risk ticket queue:

Write predicted P0/P1 results into the ticket-priority field and build a high-risk repair queue. Dispatchers can filter by risk level, fault category, and responsible department so critical tickets are not buried in ordinary repair requests.

Fault knowledge accumulation:

Continuously analyze relationships among fault category, equipment object, department, and risk level to build a text-based fault profile for equipment management. The same foundation can later support keyword extraction, equipment entity recognition, and maintenance knowledge-base construction.

Human-in-the-loop labeling:

Route low-confidence predictions, inconsistent three-task outputs, and high-risk misclassifications to manual review. Corrected samples can be fed back into the training set to improve label consistency and adapt the model to enterprise-specific language patterns.

### 5.4 Next Evolution

1. Add authorized and anonymized real enterprise work-order samples and build a validation set closer to production usage.
2. Improve risk-level annotation rules, especially for P0/P1 high-risk recall.
3. Add confusion matrices, confidence distribution analysis, and low-confidence sample review to locate misclassification boundaries.
4. Compare MacBERT, RoBERTa-wwm-ext, and other Chinese pretrained models, including full fine-tuning, frozen-encoder training, and lightweight distillation.
5. Build a low-confidence human review and active-learning loop so corrected samples can continuously return to the training data.

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

Run each step manually:

```powershell
python scripts\step2_dataset_eda.py --input data\full\chemical_repair_text_dataset_cn.csv --report artifacts\reports\eda_report.json
python scripts\step3_clean_and_split.py --input data\full\chemical_repair_text_dataset_cn.csv --output-dir data\processed\splits --labels data\processed\labels.json --report artifacts\reports\split_report.json
python scripts\step4_model_training.py --train data\processed\splits\train.csv --val data\processed\splits\val.csv --labels data\processed\labels.json --model-dir artifacts\models\baseline --backend naive_bayes --max-train-samples 2000
python scripts\step5_evaluate_and_predict.py evaluate --model-dir artifacts\models\baseline --data data\processed\splits\test.csv --report artifacts\reports\eval_report.json --predictions artifacts\reports\predictions.csv
python scripts\step5_evaluate_and_predict.py predict --model-dir artifacts\models\baseline --text "空压机运行中压力波动明显，主线节拍受到影响，请安排检修。"
```

To rebuild the full CSV from a local raw TXT/TSV source, run:

```powershell
python scripts\step1_convert_to_csv.py --input data\raw\chemical_repair_text_dataset_cn.txt --output data\full\chemical_repair_text_dataset_cn.csv
```

The public `industrial-fault-go`, Step 5 evaluation, and single-text prediction workflow currently use the `naive_bayes` backend so the pipeline can run without GPU or local pretrained-model cache. The BERT backend keeps the Step 4 training entry; unified evaluation and inference can be connected after the model artifact format is finalized.

---

## 0x07. Project Structure

```text
industrial-fault-text-classifier/
├── configs/
├── data/
│   ├── full/
│   │   └── chemical_repair_text_dataset_cn.csv
│   ├── raw/                         # Local raw source files, ignored by Git
│   └── samples/
├── docs/
├── scripts/
├── src/
│   └── industrial_fault_classifier/
├── README.md
├── README_EN.md
└── pyproject.toml
```

---

## 0x08. Version Notes

The current version focuses on repository restructuring, public CSV consolidation, step-based scripts, the baseline training loop, and README documentation. The BERT multi-task training entry is retained. A complete BERT evaluation and inference workflow can be connected after the model artifact format, evaluation interface, and serving interface are finalized.
