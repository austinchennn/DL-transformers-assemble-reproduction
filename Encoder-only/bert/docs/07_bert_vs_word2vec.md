# 07 · BERT vs Word2Vec：同样是“猜词”，差在哪里

Word2Vec（Mikolov et al., 2013）和 BERT 的 MLM 都在做“根据上下文猜一个词”，都源自同一个语言学假设。但两者学到的东西、能力上限完全不同。

## 1. 共同点

| 方面 | 说明 |
| --- | --- |
| **理论基础** | 都基于**分布式假设**（Distributional Hypothesis）：“一个词的含义由它周围的词决定”（Firth, 1957）。 |
| **自监督** | 都不需要人工标注，从原始文本中自动构造“上下文 → 目标词”的训练信号。 |
| **双向上下文** | 都同时利用目标词**左右两侧**的词（不像 GPT 只看左边）。 |
| **任务形式像完形填空** | Word2Vec 的 CBOW 用周围词预测中心词；BERT 的 MLM 用未被遮住的词预测被遮住的词。 |
| **输出层是词表上的分类** | 都是在整个词表上做预测，都需要处理大词表带来的计算开销。 |
| **学到的都是向量表示** | 训练的目的都不是“猜词”本身，而是把词映射到稠密向量，猜词只是训练信号。 |

可以把 **CBOW 看作极度简化的 MLM**：上下文窗口很小、编码器只是把词向量求平均、每次只遮一个词。

## 2. 核心区别：静态词向量 vs 上下文相关表示

### 静态 vs 动态

- **Word2Vec**：训练完后得到一张查找表，每个词**只有一个固定向量**，与它出现在哪个句子无关。
- **BERT**：同一个词在不同句子中的输出向量**不同**，由整句话经过多层自注意力计算得到。

```
我今天吃了一个 苹果。    → Word2Vec: 两句中 v(苹果) 相同
苹果 发布了新款手机。    → BERT:     h(苹果) 一个偏“水果”，一个偏“公司”
```

英文经典例子：`bank` 在 “river bank” 和 “bank account” 中，Word2Vec 给出同一个向量，只能是两种词义的混合；BERT 给出两个明显不同的向量。**多义词问题**是 BERT 相对 Word2Vec 最直观的改进。

### 对比总表

| 维度 | Word2Vec | BERT (MLM) |
| --- | --- | --- |
| 发表时间 | 2013 | 2018 |
| 模型 | 浅层：一个 Embedding 矩阵 + 一个输出矩阵，**没有隐藏层非线性** | 深层：12/24 层 Transformer Encoder |
| 参数量 | 词表 × 维度 × 2，约几百万到几亿，几乎全部是词向量 | 110M / 340M，大部分在编码器 |
| 上下文范围 | 固定小窗口（通常 ±5 个词） | 整个序列（最多 512 个 token） |
| 上下文如何组合 | CBOW：直接**平均**，忽略词序；Skip-gram：成对预测 | 多层自注意力，考虑**词序**（位置编码）和词与词之间的交互 |
| 训练目标 | CBOW：上下文 → 中心词；Skip-gram：中心词 → 每个上下文词 | 遮住 15% 的 token，预测原词；原论文另加 NSP |
| 每个句子遮多少 | 每次只预测一个位置，窗口滑动遍历所有词 | 一次随机遮 15%，同时预测 |
| 输出的计算 | Negative Sampling / Hierarchical Softmax 近似 | 完整 softmax（词表 3 万，GPU 上可以承受） |
| 词表单位 | 整词，OOV 无法表示（FastText 用字符 n-gram 改进） | WordPiece 子词，几乎没有 OOV |
| 产物 | **一张词向量表** | **一个编码器模型**（输入句子，输出每个 token 的向量） |
| 词向量是否随上下文变化 | 否（静态） | 是（上下文相关） |
| 多义词 | 一个词一个向量，多个词义混在一起 | 不同语境给出不同向量 |
| 下游使用方式 | 作为特征 / 下游模型的 Embedding 层初始化，下游仍需自己的 RNN/CNN | 加一个小输出层，整个模型一起微调 |
| 训练成本 | 单机 CPU 几小时 | 多个 TPU 数天 |
| 推理成本 | 查表，几乎为零 | 每个句子都要过一遍完整网络 |

## 3. 训练目标的公式对比

### Word2Vec Skip-gram（负采样）

对中心词 $w$ 和窗口内的上下文词 $c$，用 $k$ 个负样本 $c'_i$：

$$
\mathcal{L} = -\log\sigma\big(u_c^\top v_w\big) - \sum_{i=1}^{k}\log\sigma\big(-u_{c'_i}^\top v_w\big)
$$

$v_w$ 和 $u_c$ 都是查表得到的向量，打分就是**一个点积**。

### Word2Vec CBOW

$$
\bar v = \frac{1}{2m}\sum_{-m\le j\le m,\, j\ne 0} v_{w_{t+j}},\qquad
\mathcal{L} = -\log \frac{\exp(u_{w_t}^\top \bar v)}{\sum_{w'}\exp(u_{w'}^\top \bar v)}
$$

上下文向量是简单平均：**“狗咬人”和“人咬狗”的表示完全一样**。

### BERT MLM

$$
h = \text{TransformerEncoder}^{(N)}\big(\tilde x\big),\qquad
\mathcal{L} = -\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}\log\frac{\exp\big(e_{x_i}^\top f(h_i)\big)}{\sum_{w'}\exp\big(e_{w'}^\top f(h_i)\big)}
$$

形式上同样是“上下文表示 · 目标词向量”的 softmax。区别在于：Word2Vec 的上下文表示是**平均出来的**，BERT 的 $h_i$ 是**一个深层网络算出来的**，可以表达语法结构、远距离依赖和词序。

有意思的是，BERT 输出层与输入 Embedding 共享权重（$e_{w'}$ 就是 Token Embedding），这和 Word2Vec 中 $v$（输入向量）与 $u$（输出向量）的结构几乎一样，只是 Word2Vec 中两者通常是分开的两张表。

## 4. 学到的东西有什么不同

| | Word2Vec | BERT |
| --- | --- | --- |
| 词义相似 | ✅ `king - man + woman ≈ queen` 这类类比 | ✅ |
| 一词多义 | ❌ | ✅ |
| 词序 / 句法 | ❌（CBOW 完全丢失词序） | ✅ 研究发现中间层编码了句法树信息 |
| 指代、长距离依赖 | ❌ | ✅ 部分注意力头专门负责指代 |
| 句子级语义 | ❌ 只能把词向量平均，效果有限 | ✅（NSP / 微调后） |

## 5. 演进脉络

```
Word2Vec / GloVe (2013-14)   静态词向量，一词一向量
        ↓  问题：一词多义、不考虑语境
ELMo (2018.02)               双向 LSTM，上下文相关词向量，作为特征使用
        ↓  问题：两个单向模型浅层拼接；下游仍需专门模型
GPT (2018.06)                Transformer Decoder 预训练 + 微调，但只看左边
        ↓  问题：单向
BERT (2018.10)               深层双向 Transformer + MLM，预训练 + 微调
```

可以这样理解：**BERT 把 Word2Vec 的“根据上下文猜词”保留下来，把“上下文编码器”从“求平均”换成了深层 Transformer，并且把训练好的整个编码器（而不只是词向量表）交给下游任务**。

## 6. 什么时候还会用 Word2Vec

- 资源极度受限（嵌入式设备、超低延迟），查表几乎零成本；
- 非文本场景：推荐系统中的 item2vec、图上的 DeepWalk / node2vec，都沿用了 Skip-gram 的思路；
- 作为快速的基线或者用于词级别的分析。

## 参考

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013. arXiv:1301.3781
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality*, NeurIPS 2013
- Peters et al., *Deep contextualized word representations* (ELMo), NAACL 2018
- Devlin et al., *BERT*, NAACL 2019
- Tenney et al., *BERT Rediscovers the Classical NLP Pipeline*, ACL 2019
