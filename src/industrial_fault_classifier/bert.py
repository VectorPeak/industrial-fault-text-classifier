"""BERT full fine-tuning entry for the multi-task classifier."""

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
    """Fine-tune the full BERT encoder and three task heads."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    class RepairDataset(Dataset):
        """Torch dataset wrapper around normalized repair records."""

        def __init__(self, records: list[Record]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> Record:
            return self.records[index]

    class MultiTaskBertClassifier(nn.Module):
        """Shared BERT encoder with one linear classification head per task."""

        def __init__(self) -> None:
            super().__init__()
            # The encoder is not frozen; optimizer below updates full BERT weights.
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.heads = nn.ModuleDict(
                {
                    task: nn.Linear(hidden_size, len(schema["tasks"][task]["label2id"]))
                    for task in TASKS
                }
            )

        def forward(self, input_ids, attention_mask):
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                # Some encoders do not expose pooler_output; use masked mean pooling.
                mask = attention_mask.unsqueeze(-1)
                pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = self.dropout(pooled)
            return {task: head(pooled) for task, head in self.heads.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MultiTaskBertClassifier().to(resolved_device)
    # model.parameters() includes encoder and heads, so this is full fine-tuning.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    weights = loss_weights or {task: 1.0 for task in TASKS}

    def collate(batch: list[Record]):
        """Tokenize raw text and encode task labels for one batch."""
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

    train_loader = DataLoader(RepairDataset(train_records), batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(RepairDataset(val_records), batch_size=batch_size, shuffle=False, collate_fn=collate)

    for epoch in range(epochs):
        # Train for one epoch, then run a lightweight validation accuracy check.
        model.train()
        total_loss = 0.0
        for encoded, labels in train_loader:
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            labels = {key: value.to(resolved_device) for key, value in labels.items()}
            logits = model(**encoded)
            loss = sum(weights.get(task, 1.0) * criterion(logits[task], labels[task]) for task in TASKS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        val_accuracy = _quick_bert_accuracy(model, val_loader, resolved_device)
        print(f"epoch={epoch + 1} train_loss={total_loss / max(1, len(train_loader)):.6f} val_avg_accuracy={val_accuracy:.6f}")

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    """Compute average accuracy across the three BERT task heads."""
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
                correct += int((logits[task].argmax(dim=-1) == labels[task]).sum().detach().cpu())
    model.train()
    return correct / total if total else 0.0
