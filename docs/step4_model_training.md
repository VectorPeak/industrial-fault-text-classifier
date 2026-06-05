# step4_model_training 技术说明
# 多任务文本分类模型训练

## 0x01. 这个文件是做什么的？

Step 4 负责训练报修文本多任务分类模型。输入是一条报修文本，输出是三个结构化标签：

```text
text -> fault_category / risk_level / department
```

项目提供两条模型路线：

1. 轻量 baseline：字符 n-gram + 多任务 Naive Bayes。
2. BERT 路线：共享中文 BERT 编码器 + 三个分类头。

baseline 适合快速验证数据和工程闭环；BERT 适合在 GPU 环境和较完整训练数据上进行正式语义建模。

---

## 0x02. 整体流程图

```text
train.csv / val.csv
    |
    v
读取 labels.json
    |
    v
选择 backend
    |
    +-- naive_bayes
    |      |
    |      +-- 字符 n-gram 特征
    |      +-- 三个朴素贝叶斯分类器
    |      +-- 保存 model.pkl
    |
    +-- bert
           |
           +-- AutoTokenizer
           +-- BERT shared encoder
           +-- fault/risk/department 分类头
           +-- 保存 model.pt 和 tokenizer
```

---

## 0x03. 模型路线说明

### 3.1 baseline：字符 n-gram + Naive Bayes

baseline 使用字符级 n-gram 表示中文报修文本，默认范围为：

```text
ngram_range = (1, 2)
```

即同时使用单字和双字片段。这样做不依赖中文分词，也能捕捉“轴承”“阀位”“联锁”“温升”等短语线索。

训练时，每个任务训练一个分类器：

```text
同一组文本特征
    |
    +-- fault_category classifier
    +-- risk_level classifier
    +-- department classifier
```

baseline 的优势是依赖少、训练快、适合 CI 和 smoke test；局限是对上下文组合和隐含风险语义的建模能力有限。

### 3.2 BERT：共享编码器 + 三任务分类头

BERT 路线使用 `bert-base-chinese` 作为默认编码器。模型结构为：

```text
报修文本
    |
    v
BERT tokenizer
    |
    v
BERT encoder
    |
    v
pooled text representation
    |
    +-- fault_category head
    +-- risk_level head
    +-- department head
```

三个任务共享同一个文本语义表示，再通过不同分类头学习各自边界。这种结构适合报修文本场景，因为故障类别、风险等级和处理部门本身存在业务关联。

### 3.3 全量微调

BERT 路线不是只训练分类头，而是对 encoder 与分类头共同训练：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

`model.parameters()` 包含 BERT 编码器与三个分类头，因此属于全量微调。该方案表达能力更强，但训练成本也更高，需要 GPU 或较长训练时间。

### 3.4 损失函数

三任务训练的总损失可以理解为：

```text
total_loss =
    fault_loss * fault_weight
  + risk_loss * risk_weight
  + department_loss * department_weight
```

当前实现默认三个任务权重均为 1.0。后续如果更关注 P0/P1 高风险识别，可以提高风险等级任务权重，或在数据层面增强高风险样本。

---

## 0x04. 运行命令

训练轻量 baseline：

```powershell
python scripts\step4_model_training.py `
  --train data\processed\splits\train.csv `
  --val data\processed\splits\val.csv `
  --labels data\processed\labels.json `
  --model-dir artifacts\models\baseline `
  --backend naive_bayes
```

限制训练样本数，用于快速调试：

```powershell
python scripts\step4_model_training.py `
  --train data\processed\splits\train.csv `
  --val data\processed\splits\val.csv `
  --labels data\processed\labels.json `
  --model-dir artifacts\models\baseline_debug `
  --backend naive_bayes `
  --max-train-samples 5000
```

训练 BERT：

```powershell
python scripts\step4_model_training.py `
  --train data\processed\splits\train.csv `
  --val data\processed\splits\val.csv `
  --labels data\processed\labels.json `
  --model-dir artifacts\models\bert `
  --backend bert `
  --model-name bert-base-chinese `
  --epochs 3 `
  --batch-size 16 `
  --learning-rate 2e-5 `
  --max-length 96
```

---

## 0x05. 输出产物

| backend | 产物 | 说明 |
|---------|------|------|
| `naive_bayes` | `model.pkl` | baseline 模型 |
| `naive_bayes` | `metadata.json` | backend、训练规模和参数 |
| `bert` | `model.pt` | BERT 权重与标签结构 |
| `bert` | `tokenizer/` | tokenizer 文件 |
| 通用 | `labels.json` | 推理阶段需要的标签映射 |

所有模型产物默认写入 `artifacts/models/`，不建议提交到公开仓库。

---

## 0x06. 检查点

训练完成后，应确认：

1. `model_dir` 下存在模型文件和 `labels.json`。
2. 训练使用的 `labels.json` 与 Step 3 输出一致。
3. baseline 能完成快速闭环，BERT 用于正式语义建模。
4. BERT 训练时显存、batch size 和 `max_length` 设置匹配当前硬件。
5. 不把模型权重提交到 GitHub。
