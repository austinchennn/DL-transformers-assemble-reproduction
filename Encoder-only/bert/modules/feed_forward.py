"""逐位置前馈网络 + 残差 + LayerNorm。"""

import torch
from torch import nn

from ..config import BertConfig
from .activations import get_activation


class FeedForward(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = get_activation(config.hidden_act)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.output(self.act(self.intermediate(hidden_states)))
        return self.layer_norm(hidden_states + self.dropout(x))
