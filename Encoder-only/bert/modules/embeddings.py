"""输入嵌入 = Token Embedding + Position Embedding + Segment Embedding。"""

from typing import Optional

import torch
from torch import nn

from ..config import BertConfig


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # BERT 使用可学习的绝对位置编码，而非正弦编码
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).unsqueeze(0), persistent=False
        )

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_len]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.layer_norm(x))
