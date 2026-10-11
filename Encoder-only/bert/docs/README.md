# BERT 文档

| 文档 | 内容 |
| --- | --- |
| [01_motivation.md](01_motivation.md) | 为什么要有 BERT：单向语言模型的局限、预训练 + 微调范式 |
| [02_input.md](02_input.md) | 输入长什么样：WordPiece、特殊符号、B/L/H 符号约定、Token / Position / Segment Embedding 详解、attention mask |
| [03_architecture.md](03_architecture.md) | 完整架构：Embedding → N × Encoder Layer → Pooler / Heads，逐层张量形状 |
| [04_pretraining_tasks.md](04_pretraining_tasks.md) | 预训练任务与 loss：MLM、NSP |
| [05_training.md](05_training.md) | 训练过程：数据构造、优化器、学习率调度、原论文超参数 |
| [06_finetuning.md](06_finetuning.md) | 下游任务微调：分类、句对、序列标注、抽取式问答 |

每篇文档末尾都标注了对应的代码位置，建议对照代码阅读。
| [07_bert_vs_word2vec.md](07_bert_vs_word2vec.md) | BERT 与 Word2Vec 对比：同样是“猜词”，共同点与本质区别 |
