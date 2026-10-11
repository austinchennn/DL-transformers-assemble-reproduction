# BERT 复现

从零实现 BERT（*Pre-training of Deep Bidirectional Transformers for Language Understanding*, Devlin et al., 2018），仅依赖 PyTorch。

## 目录结构

```
bert/
├── config.py           # BertConfig：超参数、预设规模 (tiny/base/large)、JSON 序列化
├── modules/            # 可复用积木，彼此只通过 BertConfig 交互
│   ├── activations.py  # 激活函数注册表 (gelu/relu/tanh)
│   ├── embeddings.py   # Token + Position + Segment Embedding
│   ├── attention.py    # MultiHeadSelfAttention / BertAttention (Post-LN)
│   ├── feed_forward.py # FFN + 残差 + LN
│   ├── layer.py        # BertLayer = Attention + FFN
│   ├── encoder.py      # N × BertLayer
│   └── pooler.py       # [CLS] -> Linear + Tanh
├── model.py            # BertPreTrainedModel（初始化 / save / load）、BertModel 主干
├── heads.py            # MLM / NSP / 分类头 + 组合模型（BertForPreTraining 等）
├── tokenizer.py        # BasicTokenizer + WordPiece（含简化版词表训练）
├── data/               # 语料读取、NSP 句对构造、MLM 掩码 (80/10/10)、动态 padding
├── training/           # AdamW 参数分组、线性 warmup 调度、通用 Trainer
├── scripts/pretrain.py # 预训练入口
├── tests/              # 单元测试
└── docs/               # 原理文档，见 docs/README.md
```

原理文档：[动机](docs/01_motivation.md) · [输入](docs/02_input.md) · [架构](docs/03_architecture.md) · [预训练任务与 Loss](docs/04_pretraining_tasks.md) · [训练过程](docs/05_training.md) · [微调](docs/06_finetuning.md) · [BERT vs Word2Vec](docs/07_bert_vs_word2vec.md)

## 设计要点

- **主干与任务头分离**：`BertModel` 只负责编码；`BertForPreTraining` / `BertForMaskedLM` / `BertForSequenceClassification` 通过组合 `BertModel + Head` 构建，新增任务只需写一个 Head。
- **Trainer 与任务解耦**：Trainer 只依赖“`model(**batch)` 返回带 `.loss` 的对象”这一约定。
- **权重共享**：MLM decoder 与 word embedding 共享权重（`tie_weights`）。
- **迁移**：`from_pretrained(strict=False)` 可将 MLM 预训练权重加载进分类模型。
- **动态掩码**：MLM 掩码在 collator 中在线生成，每个 epoch 掩码位置不同。

## 使用

以下命令在 `Encoder-only/` 目录下执行：

```bash
# 单元测试
python -m pytest bert/tests -q

# 用内置演示语料预训练 tiny 模型
python -m bert.scripts.pretrain --epochs 3

# 自定义语料（每行一句，文档间空行分隔），仅 MLM 目标
python -m bert.scripts.pretrain --corpus corpus.txt --size base --objective mlm --batch-size 32
```

```python
from bert import BertConfig, BertModel
model = BertModel(BertConfig.tiny(vocab_size=8000))
out = model(input_ids)              # out.last_hidden_state, out.pooler_output
```
