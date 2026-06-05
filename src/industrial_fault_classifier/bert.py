"""BERT 多任务分类模型的全量微调入口。
BERT full fine-tuning entry for the multi-task classifier.
"""

from __future__ import annotations

from pathlib import Path

from .constants import TASKS
from .data import Record
from .labels import encode_label


def train_bert_classifier(
    train_records: list[Record],
    val_records: list[Record],
    schema: dict,
    model_dir: str | Path,
    model_name: str = "bert-base-chinese",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 96,
    loss_weights: dict[str, float] | None = None,
    device: str | None = None,
) -> None:
    """全量微调 BERT 编码器和三个任务分类头。
    Fine-tune the full BERT encoder and three task heads.
    """
    # 仅在用户选择 BERT 后端时导入深度学习依赖，避免 baseline 路径强依赖 torch/transformers。
    # Import deep-learning dependencies only for the BERT backend so baseline runs do not require them.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    class RepairDataset(Dataset):
        """对标准化报修记录进行 Torch Dataset 封装。
        Torch dataset wrapper around normalized repair records.
        """

        def __init__(self, records: list[Record]) -> None:
            self.records = records

        def __len__(self) -> int:
            """返回样本数量。
            Return the number of samples.
            """
            return len(self.records)

        def __getitem__(self, index: int) -> Record:
            """按索引返回单条记录。
            Return one record by index.
            """
            return self.records[index]

    class MultiTaskBertClassifier(nn.Module):
        """共享 BERT 编码器，并为每个任务配置一个线性分类头。
        Shared BERT encoder with one linear classification head per task.
        """

        def __init__(self) -> None:
            super().__init__()
            # 编码器不冻结；下方 optimizer 会同时更新 BERT 和三个分类头。
            # The encoder is not frozen; the optimizer below updates full BERT weights and all heads.
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            # 三个任务共享 pooled 表示，但各自学习独立分类边界。
            # The three tasks share the pooled representation but learn independent decision boundaries.
            self.heads = nn.ModuleDict(
                {
                    task: nn.Linear(hidden_size, len(schema["tasks"][task]["label2id"]))
                    for task in TASKS
                }
            )

        def forward(self, input_ids, attention_mask):
            """执行前向传播，并返回三个任务的 logits。
            Run a forward pass and return logits for the three tasks.
            """
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                # 部分编码器没有 pooler_output，使用 attention mask 做均值池化兜底。
                # Some encoders do not expose pooler_output; use masked mean pooling as fallback.
                mask = attention_mask.unsqueeze(-1)
                pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = self.dropout(pooled)
            return {task: head(pooled) for task, head in self.heads.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 用户未指定设备时，优先使用 CUDA；否则回退 CPU。
    # If the user does not specify a device, prefer CUDA and fall back to CPU.
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MultiTaskBertClassifier().to(resolved_device)
    # model.parameters() 包含 encoder 与 heads，因此这里是全量微调，不是只训练分类头。
    # model.parameters() includes encoder and heads, so this is full fine-tuning.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # loss_weights 允许提高风险等级或部门任务权重，适配业务优先级。
    # loss_weights can emphasize risk-level or department tasks according to business priority.
    weights = loss_weights or {task: 1.0 for task in TASKS}

    def collate(batch: list[Record]):
        """对一个 batch 执行 tokenizer 编码并转换三任务标签。
        Tokenize raw text and encode task labels for one batch.
        """
        encoded = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = {
            task: torch.tensor([encode_label(schema, task, item[task]) for item in batch], dtype=torch.long)
            for task in TASKS
        }
        return encoded, labels

    # 训练集打乱以降低样本顺序影响；验证集保持顺序便于复现。
    # Shuffle training data to reduce order effects; keep validation order deterministic.
    train_loader = DataLoader(RepairDataset(train_records), batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(RepairDataset(val_records), batch_size=batch_size, shuffle=False, collate_fn=collate)

    for epoch in range(epochs):
        # 每个 epoch 后执行轻量验证，快速观察三任务平均准确率趋势。
        # Train for one epoch, then run a lightweight validation accuracy check.
        model.train()
        total_loss = 0.0
        for encoded, labels in train_loader:
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            labels = {key: value.to(resolved_device) for key, value in labels.items()}
            logits = model(**encoded)
            # 多任务 loss 为三个交叉熵损失的加权和。
            # The multi-task loss is the weighted sum of three cross-entropy losses.
            loss = sum(weights.get(task, 1.0) * criterion(logits[task], labels[task]) for task in TASKS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        val_accuracy = _quick_bert_accuracy(model, val_loader, resolved_device)
        print(f"epoch={epoch + 1} train_loss={total_loss / max(1, len(train_loader)):.6f} val_avg_accuracy={val_accuracy:.6f}")

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存 state_dict、标签体系和 tokenizer 信息，便于后续接入统一评估或服务化推理。
    # Save state_dict, schema, and tokenizer information for later evaluation or serving integration.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "schema": schema,
            "model_name": model_name,
            "max_length": max_length,
            "backend": "bert",
        },
        output_dir / "model.pt",
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")


def _quick_bert_accuracy(model, loader, device) -> float:
    """计算三个 BERT 任务头的平均准确率。
    Compute average accuracy across the three BERT task heads.
    """
    import torch

    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for encoded, labels in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = {key: value.to(device) for key, value in labels.items()}
            logits = model(**encoded)
            batch_size = next(iter(labels.values())).shape[0]
            total += batch_size * len(TASKS)
            for task in TASKS:
                # 这里的平均准确率仅用于训练过程观察，不替代正式评估报告。
                # This average accuracy is for training feedback and does not replace the full evaluation report.
                correct += int((logits[task].argmax(dim=-1) == labels[task]).sum().detach().cpu())
    model.train()
    return correct / total if total else 0.0
