"""任务头与组合模型：主干 (BertModel) + 头 (Head) 的组合方式实现解耦。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .config import BertConfig
from .model import BertModel, BertPreTrainedModel
from .modules import get_activation

IGNORE_INDEX = -100


# ---------------------------------------------------------------- Heads
class MaskedLMHead(nn.Module):
    """Transform(Linear + act + LN) -> Decoder(与 word embedding 共享权重)。"""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            get_activation(config.hidden_act),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.transform(hidden_states)) + self.bias


class NextSentenceHead(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)


class ClassificationHead(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(pooled))


# ---------------------------------------------------------------- Outputs
@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    nsp_logits: Optional[torch.Tensor] = None
    mlm_loss: Optional[torch.Tensor] = None
    nsp_loss: Optional[torch.Tensor] = None


def _mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX)


# ---------------------------------------------------------------- Models
class BertForPreTraining(BertPreTrainedModel):
    """MLM + NSP 联合预训练（原论文设定）。"""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.mlm_head = MaskedLMHead(config)
        self.nsp_head = NextSentenceHead(config)
        self.init_weights()

    def tie_weights(self) -> None:
        self.mlm_head.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, mlm_labels=None, nsp_labels=None):
        out = self.bert(input_ids, attention_mask, token_type_ids)
        mlm_logits = self.mlm_head(out.last_hidden_state)
        nsp_logits = self.nsp_head(out.pooler_output)

        result = ModelOutput(logits=mlm_logits, nsp_logits=nsp_logits)
        if mlm_labels is not None and nsp_labels is not None:
            result.mlm_loss = _mlm_loss(mlm_logits, mlm_labels)
            result.nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
            result.loss = result.mlm_loss + result.nsp_loss
        return result


class BertForMaskedLM(BertPreTrainedModel):
    """仅 MLM（RoBERTa 之后的常见做法）。"""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.mlm_head = MaskedLMHead(config)
        self.init_weights()

    def tie_weights(self) -> None:
        self.mlm_head.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, mlm_labels=None, **_):
        out = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.mlm_head(out.last_hidden_state)
        loss = _mlm_loss(logits, mlm_labels) if mlm_labels is not None else None
        return ModelOutput(loss=loss, logits=logits, mlm_loss=loss)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.head = ClassificationHead(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **_):
        out = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.head(out.pooler_output)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return ModelOutput(loss=loss, logits=logits)
