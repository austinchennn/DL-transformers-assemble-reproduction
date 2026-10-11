# 04 · 预训练任务与 Loss

BERT 同时训练两个任务，总损失为两者之和：

$$
\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}
$$

## 1. Masked Language Model（MLM）

### 做法

1. 在每个序列中随机选择 **15%** 的 token（`[CLS]`、`[SEP]`、`[PAD]` 等特殊符号不参与）。
2. 对被选中的 token：
   - **80%** 替换为 `[MASK]`
   - **10%** 替换为词表中的随机 token
   - **10%** 保持不变
3. 模型在这些位置上预测**原始** token。

```
原始:   [CLS] my  dog    is  cute [SEP]
选中:              ^^^               ^^^^
输入:   [CLS] my  [MASK] is  apple [SEP]     ← dog 被换成 [MASK]，cute 被换成随机词
标签:   -100  -100 dog  -100 cute  -100
```

### 为什么不全部换成 `[MASK]`？

- **预训练与微调不匹配**：微调时输入里永远没有 `[MASK]`。如果预训练时模型只在 `[MASK]` 处学预测，它就可能只给 `[MASK]` 位置学好的表示。
- **10% 随机词**：迫使模型不能信任任何输入位置，必须为**每个**位置都保持良好的上下文表示。
- **10% 保持不变**：让表示偏向真实观察到的词。
- 随机替换只占全部 token 的 15% × 10% = 1.5%，不会明显损害语言理解能力。

### Loss

只在被选中的位置 $\mathcal{M}$ 上计算交叉熵：

$$
\mathcal{L}_{MLM} = -\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}} \log P(x_i \mid \tilde{x}),
\qquad P(\cdot \mid \tilde{x}) = \text{softmax}(W_{emb}\, f(h_i) + b)
$$

其中 $\tilde{x}$ 是被掩码后的输入，$h_i$ 是最后一层位置 $i$ 的隐状态，$f$ 是 MLM Head 的 transform，$W_{emb}$ 是与输入共享的词向量矩阵。实现上把非选中位置的标签设为 `-100`，用 `cross_entropy(ignore_index=-100)` 自动忽略。

### 代价

每个序列只有 15% 的位置产生训练信号，所以 MLM 比从左到右的语言模型**收敛得更慢**（ELECTRA 正是为了解决这个问题）。

### 静态 vs 动态掩码

原版 BERT 在数据预处理时就固定了掩码（把数据复制 10 份、每份掩码不同）。RoBERTa 改为**动态掩码**：每次喂给模型时重新随机掩码。本仓库在 collator 中在线掩码，即采用动态掩码。

## 2. Next Sentence Prediction（NSP）

### 做法

构造句对 `[CLS] A [SEP] B [SEP]`：

- **50%** 的情况 B 是语料中紧跟 A 的真实下一段 → `IsNext`（标签 0）
- **50%** 的情况 B 是从**其他文档**随机抽取的一段 → `NotNext`（标签 1）

```
IsNext:   [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
NotNext:  [CLS] the man went to the store [SEP] penguins are flightless birds [SEP]
```

用 `[CLS]` 位置经 Pooler 后的向量 $c$ 做二分类。

### Loss

$$
\mathcal{L}_{NSP} = -\log P(y \mid c),\qquad P(\cdot \mid c) = \text{softmax}(W_{nsp}\, c + b)
$$

### NSP 有用吗？

- 原论文消融实验显示去掉 NSP 会让 QNLI、MNLI、SQuAD 的效果变差。
- 但 **RoBERTa** 发现：去掉 NSP、改用连续的长文本片段训练，效果反而**更好**。原因可能是 NotNext 样本来自不同文档，模型只需判断**主题**是否一致就能做对，任务太简单。
- **ALBERT** 把它换成了 **SOP**（Sentence Order Prediction）：负样本是把同一文档中的两段顺序颠倒，迫使模型学习句间连贯性。

本仓库同时提供 `BertForPreTraining`（MLM + NSP，原论文设定）和 `BertForMaskedLM`（只用 MLM，RoBERTa 设定），通过 `--objective` 切换。

## 3. 训练过程中的指标

- MLM loss 初始约为 $\ln V$（$V=30522$ 时约 10.3），之后持续下降。
- NSP 准确率很快就会到 95% 以上。
- 常用 **MLM 准确率**（被掩码位置预测正确的比例）或 **perplexity** $= e^{\mathcal{L}_{MLM}}$ 衡量预训练效果。

## 代码对应

- 15% 选择 + 80/10/10 替换：[data/masking.py](../data/masking.py)
- NSP 句对构造：[data/dataset.py](../data/dataset.py)
- MLM / NSP Head 与 loss：[heads.py](../heads.py)
