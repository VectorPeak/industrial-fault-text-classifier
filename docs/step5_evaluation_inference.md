# step5_evaluation_inference 技术说明
# 多任务评估与单条报修文本推理

## 0x01. 这个文件是做什么的？

Step 5 负责模型训练后的离线评估与单条文本推理。它回答两个问题：

1. 模型在测试集上对三个任务分别表现如何。
2. 给定一条新的报修文本，模型会输出什么故障大类、风险等级和处理部门。

对维修派单项目来说，Step 5 不能只看一个总准确率。不同任务的业务代价不同，高风险漏判和处理部门误派通常比普通类别误分更需要关注。

---

## 0x02. 整体流程图

```text
trained model_dir
    |
    +-- labels.json
    +-- metadata.json
    +-- model.pkl / model.pt
    |
    v
读取 test.csv
    |
    v
批量预测三任务标签
    |
    v
计算 accuracy / macro-F1 / exact match
    |
    v
统计高频混淆组合
    |
    v
输出 eval_report.json 和 predictions.csv
```

单条推理流程：

```text
输入报修文本
    |
    v
加载模型与 labels.json
    |
    v
输出 fault_category / risk_level / department
```

---

## 0x03. 评估指标

### 3.1 单任务 accuracy

accuracy 表示某个任务预测正确的比例：

```text
correct / total
```

它直观，但容易被高频类别影响。因此在风险等级和处理部门这类可能不均衡的任务上，不能只看 accuracy。

### 3.2 macro-F1

macro-F1 先计算每个类别的 F1，再做平均。它不会因为某个类别样本多而完全支配结果，更适合观察长尾类别是否被模型忽略。

本项目 README 中展示的核心结果包括：

| 任务 | accuracy | macro-F1 |
|------|----------:|---------:|
| 故障大类 | 92.40% | 92.42% |
| 停机风险等级 | 85.84% | 86.09% |
| 处理部门 | 90.23% | 90.19% |

三任务平均 macro-F1 为 89.57%，用于概括整体分类能力。

### 3.3 three-task exact match

三任务全对率要求同一条样本的三个预测同时正确：

```text
fault_category 正确
AND risk_level 正确
AND department 正确
```

该指标比单任务 accuracy 更严格，也更接近真实派单场景。因为在工单系统中，故障类别、风险等级和处理部门通常会共同影响最终处理流程。

### 3.4 混淆组合统计

Step 5 会输出高频误分组合，例如：

```text
expected = 机械故障
predicted = 设备保养
count = N
```

这类统计用于定位模型边界问题。相比只看总分，混淆组合更能指导后续标注规范、样本补充和业务规则审计。

---

## 0x04. 运行命令

评估 baseline：

```powershell
python scripts\step5_evaluate_and_predict.py evaluate `
  --model-dir artifacts\models\baseline `
  --data data\processed\splits\test.csv `
  --report artifacts\reports\eval_report.json `
  --predictions artifacts\reports\predictions.csv
```

限制评估样本数：

```powershell
python scripts\step5_evaluate_and_predict.py evaluate `
  --model-dir artifacts\models\baseline `
  --data data\processed\splits\test.csv `
  --max-samples 1000
```

单条文本推理：

```powershell
python scripts\step5_evaluate_and_predict.py predict `
  --model-dir artifacts\models\baseline `
  --text "循环泵出口压力波动明显，伴随振动升高，请安排检查。"
```

使用统一 CLI：

```powershell
industrial-fault-classifier predict `
  --model-dir artifacts\models\baseline `
  --text "现场DCS阀位反馈异常，联锁信号间歇丢失。"
```

---

## 0x05. 输出产物

| 路径 | 说明 |
|------|------|
| `artifacts/reports/eval_report.json` | 三任务评估指标与混淆组合 |
| `artifacts/reports/predictions.csv` | 测试集预测结果 |

评估报告和预测明细属于运行产物，默认不提交。若需要在 README 中引用指标，应只摘录经过确认的关键结果。

---

## 0x06. 注意事项

1. 当前统一 evaluator 主要面向轻量 baseline 路线；BERT 的正式评估可按同一指标口径扩展。
2. `model_dir` 必须包含模型文件、`metadata.json` 和 `labels.json`。
3. 单条推理结果是辅助建议，不应直接替代工程师或调度人员判断。
4. 对 P0/P1 高风险误分样本，应建立单独复核清单。
5. 若测试指标异常接近 1.0，应优先检查数据泄漏、重复样本和切分方式。
