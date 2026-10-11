"""取 [CLS] 位置的隐状态，经 Linear + Tanh 得到句子级表示。"""

import torch
from torch import nn

from ..config import BertConfig


class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(hidden_states[:, 0]))
