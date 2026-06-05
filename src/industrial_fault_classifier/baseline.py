"""用于 smoke test 的轻量多任务 Naive Bayes baseline。
Lightweight multi-task Naive Bayes baseline for smoke tests.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from .constants import TASKS
from .data import Record
from .labels import task_labels


def tokenize(text: str, ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    """将中文报修文本切分为字符 n-gram 特征。
    Tokenize Chinese repair text into character n-grams.
    """
    # 中文报修文本不依赖外部分词器，字符 n-gram 更适合作为轻量可复现 baseline。
    # Chinese repair text can use character n-grams as a lightweight, reproducible baseline without tokenizers.
    chars = [char for char in text.strip() if not char.isspace()]
    tokens: list[str] = []
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        if n <= 0:
            continue
        # 生成 1-gram、2-gram 等连续字符片段，用于捕捉短故障词和局部搭配。
        # Generate contiguous character spans to capture short fault terms and local phrases.
        tokens.extend("".join(chars[index : index + n]) for index in range(0, max(0, len(chars) - n + 1)))
    # 空文本理论上会被清洗掉，这里保留兜底 token，避免预测阶段除零或空特征。
    # Empty text should be removed during cleaning, but a fallback token keeps prediction robust.
    return tokens or ["<EMPTY>"]


class MultitaskNaiveBayes:
    """为每个预测任务训练一个多项式 Naive Bayes 分类器。
    Train one multinomial Naive Bayes classifier per prediction task.
    """

    def __init__(self, alpha: float = 1.0, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.alpha = alpha
        self.ngram_range = ngram_range
        self.schema: dict | None = None
        self.class_doc_counts: dict[str, Counter[str]] = {}
        self.token_counts: dict[str, dict[str, Counter[str]]] = {}
        self.total_tokens: dict[str, Counter[str]] = {}
        self.vocab: set[str] = set()

    def fit(self, records: list[Record], schema: dict) -> "MultitaskNaiveBayes":
        """从训练记录中估计类别先验和 token 条件似然。
        Estimate class priors and token likelihoods from training records.
        """
        self.schema = schema
        # 每个任务独立维护类别文档数、类别 token 计数和总 token 数。
        # Each task keeps independent class document counts, class-token counts, and total-token counts.
        self.class_doc_counts = {task: Counter() for task in TASKS}
        self.token_counts = {task: defaultdict(Counter) for task in TASKS}
        self.total_tokens = {task: Counter() for task in TASKS}
        self.vocab = set()

        for record in records:
            # 同一条报修文本的字符 n-gram 特征被三个任务共享。
            # The same text features are shared by all three task-specific heads.
            tokens = tokenize(record["text"], self.ngram_range)
            self.vocab.update(tokens)
            for task in TASKS:
                # 三个任务分别统计，避免部门类别与风险等级类别相互污染。
                # Counts are task-specific so department labels and risk labels never contaminate each other.
                label = record[task]
                self.class_doc_counts[task][label] += 1
                self.token_counts[task][label].update(tokens)
                self.total_tokens[task][label] += len(tokens)
        return self

    def predict_one(self, text: str) -> Record:
        """预测单条报修文本的三个结构化标签。
        Predict all three labels for one repair text.
        """
        if self.schema is None:
            raise RuntimeError("Model is not fitted.")

        tokens = tokenize(text, self.ngram_range)
        vocab_size = max(1, len(self.vocab))
        output: Record = {"text": text, "fault_category": "", "risk_level": "", "department": ""}
        for task in TASKS:
            # 在当前任务内逐类别计算朴素贝叶斯后验分数，并选取得分最高标签。
            # Compute the posterior score for each class under one task and choose the best label.
            labels = task_labels(self.schema, task)
            total_docs = sum(self.class_doc_counts[task].values())
            best_label = labels[0]
            best_score = -float("inf")
            for label in labels:
                doc_count = self.class_doc_counts[task][label]
                # alpha 平滑用于处理训练集中未出现或低频 token，避免 log(0)。
                # Alpha smoothing handles unseen or rare tokens and prevents log(0).
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
        """批量预测记录，同时保留原始输入文本。
        Predict labels for a batch of records while preserving input text.
        """
        return [self.predict_one(record["text"]) for record in records]

    def save(self, path: str | Path) -> None:
        """将 baseline 模型保存为本地 pickle 产物。
        Persist the baseline model as a local pickle artifact.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str | Path) -> "MultitaskNaiveBayes":
        """加载并校验已保存的 baseline 模型类型。
        Load and type-check a saved baseline model.
        """
        with Path(path).open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, MultitaskNaiveBayes):
            raise TypeError(f"Unexpected model type: {type(model)!r}")
        return model
