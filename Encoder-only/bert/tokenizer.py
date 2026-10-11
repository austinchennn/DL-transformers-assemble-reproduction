"""WordPiece 分词器：BasicTokenizer（清洗/切分/标点）+ 贪心最长匹配子词切分。"""

from __future__ import annotations

import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


class BasicTokenizer:
    def __init__(self, lower_case: bool = True):
        self.lower_case = lower_case

    @staticmethod
    def _is_punct(ch: str) -> bool:
        cp = ord(ch)
        if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
            return True
        return unicodedata.category(ch).startswith("P")

    @staticmethod
    def _is_cjk(ch: str) -> bool:
        cp = ord(ch)
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
                or 0xF900 <= cp <= 0xFAFF)

    def _strip_accents(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        return "".join(c for c in text if unicodedata.category(c) != "Mn")

    def tokenize(self, text: str) -> List[str]:
        if self.lower_case:
            text = self._strip_accents(text.lower())
        tokens, buf = [], []

        def flush():
            if buf:
                tokens.append("".join(buf))
                buf.clear()

        for ch in text:
            if ch.isspace():
                flush()
            elif self._is_punct(ch) or self._is_cjk(ch):  # 标点和中文字符单独成词
                flush()
                tokens.append(ch)
            elif unicodedata.category(ch).startswith("C"):
                continue
            else:
                buf.append(ch)
        flush()
        return tokens


class WordPieceTokenizer:
    def __init__(self, vocab: Dict[str, int], lower_case: bool = True, max_chars_per_word: int = 100):
        missing = [t for t in SPECIAL_TOKENS if t not in vocab]
        if missing:
            raise ValueError(f"词表缺少特殊符号：{missing}")
        self.vocab = vocab
        self.inv_vocab = {i: t for t, i in vocab.items()}
        self.basic = BasicTokenizer(lower_case)
        self.max_chars_per_word = max_chars_per_word

    # ---- 特殊符号 id ----
    pad_id = property(lambda self: self.vocab["[PAD]"])
    unk_id = property(lambda self: self.vocab["[UNK]"])
    cls_id = property(lambda self: self.vocab["[CLS]"])
    sep_id = property(lambda self: self.vocab["[SEP]"])
    mask_id = property(lambda self: self.vocab["[MASK]"])

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def special_ids(self) -> set:
        return {self.vocab[t] for t in SPECIAL_TOKENS}

    # ---- 构造 / 持久化 ----
    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "WordPieceTokenizer":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls({tok: i for i, tok in enumerate(lines) if tok}, **kwargs)

    def save(self, path: str | Path) -> None:
        tokens = [self.inv_vocab[i] for i in range(len(self.inv_vocab))]
        Path(path).write_text("\n".join(tokens) + "\n", encoding="utf-8")

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 8000, min_freq: int = 2,
              lower_case: bool = True) -> "WordPieceTokenizer":
        """简化版词表构建：特殊符号 + 全部字符（含 ## 续接形式）+ 高频整词。

        真正的 WordPiece 训练基于似然增益合并子词，这里以频率近似，足够用于复现实验。
        """
        basic = BasicTokenizer(lower_case)
        word_freq: Counter = Counter()
        for t in texts:
            word_freq.update(basic.tokenize(t))

        chars: Counter = Counter()
        for w, f in word_freq.items():
            chars[w[0]] += f
            for c in w[1:]:
                chars["##" + c] += f

        vocab_list = list(SPECIAL_TOKENS)
        for tok, _ in chars.most_common():
            if len(vocab_list) >= vocab_size:
                break
            vocab_list.append(tok)
        for w, f in word_freq.most_common():
            if len(vocab_list) >= vocab_size or f < min_freq:
                break
            if len(w) > 1:
                vocab_list.append(w)
        return cls({t: i for i, t in enumerate(vocab_list)}, lower_case=lower_case)

    # ---- 编码 / 解码 ----
    def _wordpiece(self, word: str) -> List[str]:
        if len(word) > self.max_chars_per_word:
            return ["[UNK]"]
        pieces, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                sub = word[start:end] if start == 0 else "##" + word[start:end]
                if sub in self.vocab:
                    cur = sub
                    break
                end -= 1
            if cur is None:
                return ["[UNK]"]
            pieces.append(cur)
            start = end
        return pieces

    def tokenize(self, text: str) -> List[str]:
        return [p for w in self.basic.tokenize(text) for p in self._wordpiece(w)]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def encode(self, text_a: str, text_b: Optional[str] = None, max_length: int = 128) -> Dict[str, List[int]]:
        """[CLS] A [SEP] (B [SEP])，超长时按最长优先截断。"""
        a = self.convert_tokens_to_ids(self.tokenize(text_a))
        b = self.convert_tokens_to_ids(self.tokenize(text_b)) if text_b is not None else None
        return self.build_inputs(a, b, max_length)

    def build_inputs(self, a: List[int], b: Optional[List[int]] = None, max_length: int = 128):
        a, b = list(a), list(b) if b is not None else None
        budget = max_length - (3 if b is not None else 2)
        while len(a) + (len(b) if b else 0) > budget:
            (a if not b or len(a) >= len(b) else b).pop()
        ids = [self.cls_id] + a + [self.sep_id]
        types = [0] * len(ids)
        if b is not None:
            ids += b + [self.sep_id]
            types += [1] * (len(b) + 1)
        return {"input_ids": ids, "token_type_ids": types, "attention_mask": [1] * len(ids)}

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            if skip_special and i in self.special_ids:
                continue
            tok = self.inv_vocab.get(i, "[UNK]")
            if tok.startswith("##") and out:
                out[-1] += tok[2:]
            else:
                out.append(tok)
        return " ".join(out)
