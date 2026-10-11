"""单个 Transformer Encoder 层。"""

from torch import nn

from ..config import BertConfig
from .attention import BertAttention
from .feed_forward import FeedForward


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states, attention_mask=None, return_attention: bool = False):
        hidden_states, probs = self.attention(hidden_states, attention_mask, return_attention)
        return self.feed_forward(hidden_states), probs
