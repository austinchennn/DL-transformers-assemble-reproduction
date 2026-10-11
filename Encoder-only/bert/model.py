"""BERT 主干与预训练基类。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn

from .config import BertConfig
from .modules import BertEmbeddings, BertEncoder, BertPooler


class BertPreTrainedModel(nn.Module):
    """所有 BERT 系列模型的基类：负责权重初始化与 save / load。"""

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "model.pt"

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

    def init_weights(self) -> None:
        self.apply(self._init_module)
        self.tie_weights()

    def _init_module(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def tie_weights(self) -> None:
        """子类如有输出层需要与输入 Embedding 共享权重，重写此方法。"""

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory / self.CONFIG_NAME)
        torch.save(self.state_dict(), directory / self.WEIGHTS_NAME)

    @classmethod
    def from_pretrained(cls, directory: str | Path, strict: bool = False, **kwargs):
        """加载权重；strict=False 时允许只加载主干（如从预训练模型初始化分类模型）。"""
        directory = Path(directory)
        config = BertConfig.load(directory / cls.CONFIG_NAME)
        model = cls(config, **kwargs)
        state = torch.load(directory / cls.WEIGHTS_NAME, map_location="cpu")
        model.load_state_dict(state, strict=strict)
        model.tie_weights()
        return model


@dataclass
class BertModelOutput:
    last_hidden_state: torch.Tensor
    pooler_output: Optional[torch.Tensor]
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None


class BertModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()

    @staticmethod
    def build_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """(B, L) 的 0/1 掩码 -> (B, 1, 1, L) 的加性掩码。"""
        mask = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> BertModelOutput:
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        x = self.embeddings(input_ids, token_type_ids)
        ext_mask = self.build_attention_mask(attention_mask, x.dtype)
        enc = self.encoder(x, ext_mask, output_hidden_states, output_attentions)
        pooled = self.pooler(enc.last_hidden_state) if self.pooler is not None else None
        return BertModelOutput(enc.last_hidden_state, pooled, enc.hidden_states, enc.attentions)
