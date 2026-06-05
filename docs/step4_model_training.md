# Step 4 Model Training 技术说明

## 0x01. 目标

训练面向三任务输出的文本分类模型，将报修文本同时映射到故障大类、停机风险等级和处理部门。

## 0x02. 输入与输出

| 项目 | 路径示例 | 说明 |
|------|----------|------|
| 训练集 | `data/processed/splits/train.csv` | Step 3 输出 |
| 验证集 | `data/processed/splits/val.csv` | Step 3 输出 |
| 标签映射 | `data/processed/labels.json` | Step 3 输出 |
| 模型目录 | `artifacts/models/baseline` | 本地模型产物，默认不提交 |

## 0x03. 模型路线

| 后端 | 适用场景 | 特点 |
|------|----------|------|
| `naive_bayes` | 工程闭环、快速调试、无 GPU 环境 | 字符 n-gram，多任务分别建模 |
| `bert` | 正式语义建模 | 共享 BERT 编码器，三个分类头；当前保留训练入口 |

BERT 后端使用共享编码器提取文本表示，然后分别输出三个任务的 logits。损失函数为三个交叉熵损失的加权和，风险等级和处理部门可设置更高权重。

## 0x04. 运行命令

轻量 baseline：

```powershell
python scripts\step4_model_training.py --train data\processed\splits\train.csv --val data\processed\splits\val.csv --labels data\processed\labels.json --model-dir artifacts\models\baseline --backend naive_bayes --max-train-samples 2000
```

BERT 后端：

```powershell
python scripts\step4_model_training.py --train data\processed\splits\train.csv --val data\processed\splits\val.csv --labels data\processed\labels.json --model-dir artifacts\models\bert --backend bert --model-name bert-base-chinese --epochs 3 --batch-size 16 --max-length 96
```

## 0x05. 产物

训练产物包含模型文件、标签映射和元数据。模型目录已加入 `.gitignore`，如需发布模型，应先确认数据授权、模型大小和敏感信息风险。

当前 Step 5 的统一评估与推理闭环针对 `naive_bayes` 后端。BERT 后端训练入口用于后续效果实验，评估和推理接口需要在模型保存格式稳定后继续补齐。
