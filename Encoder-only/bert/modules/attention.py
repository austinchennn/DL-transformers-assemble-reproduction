"""多头自注意力。"""

import math
from typing import Optional

import torch
from torch import nn

from ..config import BertConfig


class MultiHeadSelfAttention(nn.Module):
    """Scaled Dot-Product Attention 的多头版本（不含输出投影后的残差）。"""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, H) -> (B, heads, L, head_dim)
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Args:
            hidden_states: (B, L, H)
            attention_mask: 加性掩码，可广播到 (B, heads, L, L)，屏蔽位置为一个很大的负数
        """
        q = self._split_heads(self.query(hidden_states))
        k = self._split_heads(self.key(hidden_states))
        v = self._split_heads(self.value(hidden_states))

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = self.dropout(scores.softmax(dim=-1))

        context = probs @ v  # (B, heads, L, head_dim)
        b, _, l, _ = context.shape
        context = context.transpose(1, 2).reshape(b, l, -1)
        return (context, probs) if return_attention else (context, None)


class BertAttention(nn.Module):
    """自注意力 + 输出投影 + 残差 + LayerNorm（Post-LN）。"""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, return_attention: bool = False):
        context, probs = self.self_attn(hidden_states, attention_mask, return_attention)
        out = self.layer_norm(hidden_states + self.dropout(self.output(context)))
        return out, probs
