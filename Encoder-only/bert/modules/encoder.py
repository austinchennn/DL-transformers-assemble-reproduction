"""N 层 BertLayer 堆叠。"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from ..config import BertConfig
from .layer import BertLayer


@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = nn.ModuleList(BertLayer(config) for _ in range(config.num_hidden_layers))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> EncoderOutput:
        all_hidden = [hidden_states] if output_hidden_states else None
        all_attn = [] if output_attentions else None

        for layer in self.layers:
            hidden_states, probs = layer(hidden_states, attention_mask, output_attentions)
            if output_hidden_states:
                all_hidden.append(hidden_states)
            if output_attentions:
                all_attn.append(probs)

        return EncoderOutput(hidden_states, all_hidden, all_attn)
