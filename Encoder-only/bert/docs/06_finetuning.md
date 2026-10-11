# 06 · 下游任务微调

预训练完成后，在 BERT 顶部加一个很小的输出层，**所有参数一起**在有标注数据上训练。

## 1. 四类典型任务

| 任务类型 | 输入 | 用哪个输出 | 输出层 | 例子 |
| --- | --- | --- | --- | --- |
| 单句分类 | `[CLS] A [SEP]` | `[CLS]` / pooler | `Linear(H, C)` | 情感分析 SST-2 |
| 句对分类 | `[CLS] A [SEP] B [SEP]` | `[CLS]` / pooler | `Linear(H, C)` | 自然语言推断 MNLI、复述 QQP |
| 序列标注 | `[CLS] A [SEP]` | 每个 token 的隐状态 | `Linear(H, 标签数)` | 命名实体识别 CoNLL-2003 |
| 抽取式问答 | `[CLS] 问题 [SEP] 段落 [SEP]` | 段落中每个 token 的隐状态 | 两个向量 $S, E \in \mathbb{R}^H$ | SQuAD |

### 分类 loss

$$
\mathcal{L} = -\log \text{softmax}(W\,c + b)_y,\qquad c = \text{pooler\_output}
$$

### 抽取式问答

对段落中每个位置 $i$，起点概率 $P^{start}_i = \text{softmax}_i(S \cdot h_i)$，终点概率同理用 $E$。loss 为起点和终点的交叉熵之和；预测时选择 $j \ge i$ 中使 $S\cdot h_i + E\cdot h_j$ 最大的区间。

## 2. 微调超参数（原论文推荐）

| 超参数 | 搜索范围 |
| --- | --- |
| batch size | 16, 32 |
| 学习率 | 5e-5, 3e-5, 2e-5 |
| epoch 数 | 2, 3, 4 |
| dropout | 0.1 |

学习率比预训练小得多，以免破坏预训练学到的知识。小数据集上微调结果方差较大，建议多个随机种子取最好。

## 3. 在本仓库中微调

预训练权重的参数名以 `bert.` 开头，分类模型的主干也叫 `bert`，因此可以直接加载；分类头是新的、随机初始化的（`strict=False` 允许缺失）：

```python
from bert import BertForSequenceClassification, WordPieceTokenizer

tokenizer = WordPieceTokenizer.from_file("outputs/bert/vocab.txt")
model = BertForSequenceClassification.from_pretrained("outputs/bert")

enc = tokenizer.encode("the movie was great", max_length=128)
# 组 batch 后：
out = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
out.loss.backward()
```

训练循环可直接复用 `training.Trainer`——它只要求 `model(**batch)` 返回带 `.loss` 的对象，因此 batch 里的标签字段名为 `labels` 即可。

如果预训练用的是 `BertForMaskedLM`（没有 Pooler），加载到分类模型时 Pooler 会随机初始化，这是正常的。

## 4. 用作特征 / Embedding

不微调也可以把 BERT 当作编码器使用：

- 取 `last_hidden_state` 作为每个 token 的上下文表示；
- 句向量常用**对非 padding token 的隐状态做平均**（mean pooling），通常比直接取 `[CLS]` 更好；
- 原始 BERT 的句向量用于语义相似度效果一般，Sentence-BERT、SimCSE、BGE 等通过对比学习进一步训练，才得到好用的 Embedding 模型。

## 代码对应

- 分类模型：`BertForSequenceClassification`，[heads.py](../heads.py)
- 加载预训练权重：`BertPreTrainedModel.from_pretrained`，[model.py](../model.py)
