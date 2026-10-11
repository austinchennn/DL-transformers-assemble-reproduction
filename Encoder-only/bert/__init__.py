"""从零复现 BERT (Devlin et al., 2018)。

模块划分：
    config      —— 超参数配置
    modules     —— 可复用的网络积木（Embedding / Attention / FFN / Layer / Encoder / Pooler）
    model       —— BertModel 主干与预训练基类
    heads       —— 任务头（MLM / NSP / 分类）及组合模型
    tokenizer   —— WordPiece 分词器
    data        —— 预训练数据构造与 MLM 掩码
    training    —— 优化器、学习率调度、Trainer
"""

from .config import BertConfig
from .model import BertModel, BertPreTrainedModel
from .heads import BertForPreTraining, BertForMaskedLM, BertForSequenceClassification
from .tokenizer import WordPieceTokenizer

__all__ = [
    "BertConfig",
    "BertModel",
    "BertPreTrainedModel",
    "BertForPreTraining",
    "BertForMaskedLM",
    "BertForSequenceClassification",
    "WordPieceTokenizer",
]
