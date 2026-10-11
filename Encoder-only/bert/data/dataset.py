"""预训练样本构造：按文档切句，生成 NSP 句对（50% IsNext / 50% NotNext）。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from ..tokenizer import WordPieceTokenizer


def read_documents(path: str | Path) -> List[List[str]]:
    """语料格式：每行一句，文档之间以空行分隔（与原始 BERT 一致）。"""
    docs, cur = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cur.append(line)
        elif cur:
            docs.append(cur)
            cur = []
    if cur:
        docs.append(cur)
    return docs


class PretrainingDataset(Dataset):
    def __init__(
        self,
        documents: List[List[str]],
        tokenizer: WordPieceTokenizer,
        max_length: int = 128,
        nsp_probability: float = 0.5,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = random.Random(seed)
        self.docs = [[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)) for s in d] for d in documents]
        self.docs = [[s for s in d if s] for d in self.docs]
        self.docs = [d for d in self.docs if d]
        if not self.docs:
            raise ValueError("语料为空")
        self.examples = self._build_examples(nsp_probability)

    def _random_segment(self, exclude_doc: int) -> List[int]:
        candidates = [i for i in range(len(self.docs)) if i != exclude_doc] or [exclude_doc]
        doc = self.docs[self.rng.choice(candidates)]
        return list(doc[self.rng.randrange(len(doc))])

    def _build_examples(self, nsp_probability: float) -> List[dict]:
        examples = []
        target = self.max_length - 3
        for di, doc in enumerate(self.docs):
            i = 0
            while i < len(doc):
                # 累积若干句子组成一个 chunk，再随机切成 A / B 两段
                chunk, length = [], 0
                while i < len(doc) and (not chunk or length + len(doc[i]) <= target):
                    chunk.append(doc[i])
                    length += len(doc[i])
                    i += 1
                split = self.rng.randint(1, len(chunk) - 1) if len(chunk) > 1 else 1
                a = [t for s in chunk[:split] for t in s]
                if len(chunk) > 1 and self.rng.random() >= nsp_probability:
                    b, is_next = [t for s in chunk[split:] for t in s], 0
                else:
                    b, is_next = self._random_segment(di), 1
                    if len(chunk) == 1:
                        is_next = 1
                feats = self.tokenizer.build_inputs(a, b, self.max_length)
                feats["nsp_label"] = is_next  # 0 = IsNext, 1 = NotNext
                examples.append(feats)
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]
