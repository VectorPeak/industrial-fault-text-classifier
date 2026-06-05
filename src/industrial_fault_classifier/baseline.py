from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from .constants import TASKS
from .data import Record
from .labels import task_labels


def tokenize(text: str, ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    chars = [char for char in text.strip() if not char.isspace()]
    tokens: list[str] = []
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        if n <= 0:
            continue
        tokens.extend("".join(chars[index : index + n]) for index in range(0, max(0, len(chars) - n + 1)))
    return tokens or ["<EMPTY>"]


class MultitaskNaiveBayes:
    def __init__(self, alpha: float = 1.0, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.alpha = alpha
        self.ngram_range = ngram_range
        self.schema: dict | None = None
        self.class_doc_counts: dict[str, Counter[str]] = {}
        self.token_counts: dict[str, dict[str, Counter[str]]] = {}
        self.total_tokens: dict[str, Counter[str]] = {}
        self.vocab: set[str] = set()

    def fit(self, records: list[Record], schema: dict) -> "MultitaskNaiveBayes":
        self.schema = schema
        self.class_doc_counts = {task: Counter() for task in TASKS}
        self.token_counts = {task: defaultdict(Counter) for task in TASKS}
        self.total_tokens = {task: Counter() for task in TASKS}
        self.vocab = set()

        for record in records:
            tokens = tokenize(record["text"], self.ngram_range)
            self.vocab.update(tokens)
            for task in TASKS:
                label = record[task]
                self.class_doc_counts[task][label] += 1
                self.token_counts[task][label].update(tokens)
                self.total_tokens[task][label] += len(tokens)
        return self

    def predict_one(self, text: str) -> Record:
        if self.schema is None:
            raise RuntimeError("Model is not fitted.")

        tokens = tokenize(text, self.ngram_range)
        vocab_size = max(1, len(self.vocab))
        output: Record = {"text": text, "fault_category": "", "risk_level": "", "department": ""}
        for task in TASKS:
            labels = task_labels(self.schema, task)
            total_docs = sum(self.class_doc_counts[task].values())
            best_label = labels[0]
            best_score = -float("inf")
            for label in labels:
                doc_count = self.class_doc_counts[task][label]
                prior = math.log((doc_count + self.alpha) / (total_docs + self.alpha * len(labels)))
                denom = self.total_tokens[task][label] + self.alpha * vocab_size
                likelihood = sum(
                    math.log((self.token_counts[task][label][token] + self.alpha) / denom)
                    for token in tokens
                )
                score = prior + likelihood
                if score > best_score:
                    best_label = label
                    best_score = score
            output[task] = best_label
        return output

    def predict_many(self, records: list[Record]) -> list[Record]:
        return [self.predict_one(record["text"]) for record in records]

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str | Path) -> "MultitaskNaiveBayes":
        with Path(path).open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, MultitaskNaiveBayes):
            raise TypeError(f"Unexpected model type: {type(model)!r}")
        return model

