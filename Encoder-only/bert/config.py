"""BERT 超参数配置。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    num_labels: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) 必须能被 "
                f"num_attention_heads ({self.num_attention_heads}) 整除"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    # ---- 预设规模 ----
    @classmethod
    def base(cls, **kwargs) -> "BertConfig":
        return cls(**kwargs)

    @classmethod
    def large(cls, **kwargs) -> "BertConfig":
        defaults = dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096)
        return cls(**{**defaults, **kwargs})

    @classmethod
    def tiny(cls, **kwargs) -> "BertConfig":
        """用于调试 / 单测的小模型。"""
        defaults = dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512,
                        max_position_embeddings=128)
        return cls(**{**defaults, **kwargs})

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BertConfig":
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BertConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
