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

## 0x03. Model And Technical Selection Process

### 3.1 Stage 1: Limits Of Rule-Based Dispatch

The initial idea was to classify repair text with keyword dictionaries, fault phrases, and department mapping rules. For example, terms such as bearing, vibration, or temperature rise could indicate mechanical faults, while interlock, DCS, and valve feedback could indicate control or instrumentation issues.

The rule-based route has clear limitations:

1. Chemical-industry repair text contains many abbreviations, informal descriptions, and synonymous expressions, so rule coverage becomes expensive to maintain.
2. P0/P1 high-risk identification often depends on contextual combinations rather than isolated keywords.
3. Responsible department routing is not a one-to-one mapping from fault category; equipment object, impact scope, and risk level can change the dispatch decision.
4. Rules are interpretable, but generalization is limited and long-term iteration becomes difficult.

Conclusion: rules are useful for data audits, leakage-phrase checks, and human-review assistance, but they are not suitable as the main modeling route.

### 3.2 Stage 2: Route Comparison

Before fixing the final structure, the project compares four technical routes:

| Route | Representative Methods | Strengths | Limitations | Project Role |
|------|------|------|------|------|
| Keyword rules | Dictionaries, regular expressions, manual mapping tables | Interpretable, low deployment cost, easy to audit | High maintenance cost and weak coverage of implicit risk | Used for audit and leakage checks, not as the main model |
| Traditional machine learning | TF-IDF / character n-gram + Naive Bayes / Linear SVM | Fast training, few dependencies, good for engineering validation | Limited ability to model long-range semantics and contextual combinations | Default public baseline |
| Pretrained language models | Chinese BERT / MacBERT / RoBERTa-wwm-ext | Strong contextual semantics for complex repair text | Higher training cost; needs GPU and model cache | Formal semantic modeling route |
| Large-language-model classification | Prompt classification, few-shot classification, external inference APIs | Fast cold start and rich explanations | Cost, stability, data security, and batch inference control need extra review | Possible future quality-check or review assistant |

Final decision: keep a dual-route design with a lightweight baseline and a BERT multi-task training entry. The baseline verifies data, scripts, and evaluation flow quickly; the BERT route is reserved for formal semantic modeling on authorized enterprise data.

### 3.3 Stage 3: From Single-Task Classification To Multi-Task Modeling

A straightforward first implementation would train three independent classifiers: one for fault category, one for downtime risk level, and one for responsible department. This is simple, but it ignores the business relationship among the three tasks.

The project therefore maps one repair text to three structured labels:

```text
Repair text
    |
Shared text features or shared semantic representation
    |
fault_category / risk_level / department
```

This design is used because:

1. Fault category and responsible department are strongly related.
2. Risk level affects dispatch priority and downstream human-review strategy.
3. A three-task joint output is closer to real dispatch decisions than one isolated label.
4. The BERT route can reuse one shared encoder and learn task-specific decision boundaries through three heads.

The public baseline uses the same character n-gram feature logic and trains lightweight classifiers for the three tasks. The BERT entry uses a shared Chinese BERT encoder with three classification heads and performs full fine-tuning of the encoder and heads.

### 3.4 Stage 4: Dataset Split Strategy

If the dataset is stratified only by one label such as `fault_category`, the joint distribution of `risk_level` and `department` may drift across train, validation, and test sets. For a multi-task dispatch system, this directly affects evaluation reliability.

Step 3 therefore stratifies by the combined three-task label:

```text
stratify_key = fault_category + risk_level + department
```

The goal is not superficial balance on a single task, but preserving real dispatch combinations across `train / val / test` so evaluation better reflects end-to-end usage.

### 3.5 Stage 5: From Accuracy To Risk-Oriented Evaluation

It is tempting to evaluate the model only with overall accuracy. In chemical-industry repair dispatch, however, different mistakes carry different business costs. Missing a P0/P1 high-risk ticket is more serious than misclassifying an ordinary P2/P3 ticket. Department misrouting creates transfer and waiting costs. Any one of the three task errors may affect dispatch decisions.

The evaluation scope is therefore expanded to:

1. `accuracy`: overall task-level classification ability.
2. `macro-F1`: performance across labels without hiding low-frequency classes.
3. `P0/P1 recall`: whether high-risk work orders are identified in time.
4. `three-task exact match`: whether all three predictions are correct at the same time.
5. Confusion summaries: typical misclassification boundaries among fault category, risk level, and department.

Conclusion: this project does not optimize for one accuracy number only. The evaluation system is built around dispatch risk, review cost, and high-risk recall.

### 3.6 Stage 6: From Offline Classification To Human-In-The-Loop Operation

Offline classification only completes the first step from text to labels. In enterprise deployment, the model output should be used as dispatch support rather than a replacement for human judgment.

Recommended loop:

```text
New repair text
    |
Three-task model output
    |
Confidence and risk-level check
    |
High-risk or low-confidence samples go to human review
    |
Reviewed results return to training data
```

This design keeps final control with human reviewers in safety-critical production settings while letting the model handle frequent, repetitive, standardized triage work. With real enterprise tickets, the next priority should be low-confidence sample feedback, P0/P1 misclassification audits, and department misrouting analysis.

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

## 0x07. Key Lessons

Text structuring is the starting point of maintenance-data governance: chemical-industry repair records must first be converted into stable labels such as fault category, downtime risk level, and responsible department before dispatch optimization, risk statistics, and equipment profiling become computable.

Multi-task modeling is closer to real dispatch than isolated classification: predicting only the fault category answers what happened, while jointly predicting risk level and department also answers how urgent it is and who should handle it.

Data quality control defines the model ceiling: empty fields, duplicate samples, same-text label conflicts, and potential label leakage directly affect supervision reliability. For work-order text projects, cleaning, auditing, and stratified splitting are as important as the model architecture.

P0/P1 recall is more important than overall accuracy: in chemical production, missing a high-risk ticket is more costly than ordinary class confusion. Evaluation must therefore include P0/P1 recall, macro-F1, three-task exact match, and confusion summaries instead of accuracy alone.

The baseline validates the engineering loop: character n-gram Naive Bayes does not represent the final performance ceiling, but it has few dependencies, trains quickly, and verifies whether data loading, label mapping, splitting, training, evaluation, and inference are connected.

The BERT route is reserved for formal semantic modeling: after authorized and anonymized enterprise work orders are available, Chinese BERT, MacBERT, RoBERTa-wwm-ext, full fine-tuning, frozen-encoder training, and lightweight distillation can be compared systematically.

Human-in-the-loop operation matters more than one-off training: low-confidence samples, high-risk misclassifications, and department-rerouting cases should enter manual review, with corrected labels fed back into the training set. In the long run, stable annotation rules and continuous feedback determine deployment quality more than a single training run.

---

## 0x08. Project Structure

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

## 0x09. Version Notes

The current version focuses on repository restructuring, public CSV consolidation, step-based scripts, the baseline training loop, and README documentation. The BERT multi-task training entry is retained. A complete BERT evaluation and inference workflow can be connected after the model artifact format, evaluation interface, and serving interface are finalized.
