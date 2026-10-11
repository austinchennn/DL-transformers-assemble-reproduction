# 02 · 输入长什么样

## 1. 从文本到 token

### WordPiece 分词

BERT 使用 WordPiece 子词分词，词表大小约 30k（英文 uncased 版本为 30522）。

1. **Basic tokenization**：小写化、去除重音、按空白切分、标点单独成词、中文字符逐字切分。
2. **WordPiece**：对每个词做**贪心最长匹配**，非词首的子词加 `##` 前缀；无法匹配则为 `[UNK]`。

```
"unaffable playing"  →  un ##aff ##able play ##ing
```

子词的好处：词表有限却几乎没有 OOV，低频词可以拆成高频片段。

### 特殊符号

| 符号 | 作用 |
| --- | --- |
| `[CLS]` | 序列第一个位置，其最终隐状态作为整句表示，用于分类 / NSP |
| `[SEP]` | 句子分隔符，也放在序列末尾 |
| `[MASK]` | MLM 中替换被遮住的词 |
| `[PAD]` | 补齐到同一长度 |
| `[UNK]` | 未知词 |

## 2. 输入格式

单句：

```
[CLS] my dog is cute [SEP]
```

句对（NSP、问答、自然语言推断等）：

```
[CLS] my dog is cute [SEP] he likes play ##ing [SEP]
```

模型的一个样本由三个等长的整数序列组成：

| 字段 | 形状 | 含义 | 例子 |
| --- | --- | --- | --- |
| `input_ids` | `(B, L)` | token 在词表中的 id | `[2, 1037, 3899, ..., 3, 0, 0]` |
| `token_type_ids` | `(B, L)` | 属于句子 A 为 0，属于句子 B 为 1（segment id） | `[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]` |
| `attention_mask` | `(B, L)` | 真实 token 为 1，padding 为 0 | `[1, 1, ..., 1, 0, 0]` |

预训练时还会有：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `mlm_labels` | `(B, L)` | 被选中位置为原始 token id，其余为 `-100`（忽略） |
| `nsp_labels` | `(B,)` | 0 = IsNext，1 = NotNext |

### 截断

最大长度 $L_{max}$（原论文 512）。句对超长时，每次从**较长**的一句末尾删一个 token，直到满足 $|A| + |B| + 3 \le L_{max}$（3 = 一个 `[CLS]` + 两个 `[SEP]`）。

## 3. 符号约定：B、L、H

文档和代码注释中的张量形状统一用以下字母：

| 符号 | 全称 | 含义 | 典型值 | 代码中对应 |
| --- | --- | --- | --- | --- |
| **B** | Batch size | 一次并行处理多少个序列（样本） | 32、256 | `input_ids.size(0)` |
| **L** | Length（序列长度） | 每个序列有多少个 token，**包括** `[CLS]`、`[SEP]` 和 padding；同一个 batch 内补齐到相同长度 | 128、512 | `input_ids.size(1)` |
| **H** | Hidden size | 每个 token 用多长的向量表示，即模型宽度 | Base 768，Large 1024 | `config.hidden_size` |
| V | Vocab size | 词表大小 | 30522 | `config.vocab_size` |
| A | Attention heads | 注意力头数 | 12 | `config.num_attention_heads` |
| $d_k$ | head dim | 每个头的维度，$H / A$ | 64 | `config.head_dim` |

所以 `(B, L, H)` 读作：**B 个句子，每句 L 个 token，每个 token 一个 H 维向量**。例如 `(32, 128, 768)` 表示 32 个句子 × 128 个位置 × 768 维。

```
             H = 768 维
           ┌──────────────┐
  L=128  ┌─┤ [CLS] 的向量  │
  个位置  │ │ my    的向量  │   ← 一个句子是 (L, H) 的矩阵
         │ │ ...          │
         └─┤ [PAD] 的向量  │
           └──────────────┘
  再把 B=32 个这样的矩阵叠起来 → (B, L, H)
```

在整个 Encoder 中，**每一层的输入、输出形状都是 `(B, L, H)`**，只有注意力内部（拆成多头 `(B, A, L, d_k)`、注意力矩阵 `(B, A, L, L)`）和 FFN 内部（`(B, L, 4H)`）会临时变形。

## 4. 输入表示 = 三种 Embedding 相加

### 4.1 什么是 Embedding

`nn.Embedding(N, H)` 本质是一个 **N 行 H 列的可学习矩阵**，“embedding 一个 id”就是**取出第 id 行**（查表）。等价于用 one-hot 向量乘这个矩阵，但查表更快。矩阵的数值在训练中和其他参数一起通过反向传播更新。

BERT 有三张这样的表，分别回答三个问题：

| | 回答的问题 | 表的形状 | 用什么去查 | 查出来的形状 |
| --- | --- | --- | --- | --- |
| **Token Embedding** | 这个位置是**什么词**？ | `(V, H)` = `(30522, 768)` | `input_ids` | `(B, L, H)` |
| **Position Embedding** | 这个词在**第几个位置**？ | `(512, H)` | `[0, 1, ..., L-1]` | `(1, L, H)`，在 batch 上广播 |
| **Segment Embedding** | 这个词属于**句子 A 还是 B**？ | `(2, H)` | `token_type_ids` | `(B, L, H)` |

三者形状相同，**逐元素相加**，再经过 LayerNorm 和 Dropout，得到 Encoder 的输入 `(B, L, H)`：

$$
E = \text{Dropout}\Big(\text{LayerNorm}\big(E_{token}[x] + E_{position}[0..L-1] + E_{segment}[s]\big)\Big)
$$

```
input:      [CLS]  my   dog   is  cute [SEP]  he  likes play ##ing [SEP]
Token:      E_cls  E_my E_dog E_is E_cute E_sep E_he ...            E_sep
Segment:    E_A    E_A  E_A   E_A  E_A  E_A   E_B  E_B  E_B   E_B   E_B
Position:   E_0    E_1  E_2   E_3  E_4  E_5   E_6  E_7  E_8   E_9   E_10
              +      +    +     +   ...
          = 每个位置一个 H 维向量
```

### 4.2 Token Embedding（词嵌入）

- **作用**：把离散的 token id 变成稠密向量，表示“这是哪个词”。这是模型对词义的初始认知，语义相近的词训练后向量也相近。
- **形状**：`(V, H)`，每个 WordPiece 子词一行。BERT-Base 中有 30522 × 768 ≈ 2340 万个参数，约占全模型的 21%。
- **`[PAD]` 那一行**初始化为 0，并设置 `padding_idx`，不接受梯度更新。
- **与输出层共享**：MLM Head 最后预测词表时，直接复用这张表（weight tying），见 [03_architecture.md](03_architecture.md)。
- **局限**：它是**静态的**，同一个词不管在哪个句子里查出来都一样；上下文相关的含义要靠后面的自注意力层计算出来（对比 [07_bert_vs_word2vec.md](07_bert_vs_word2vec.md)）。

### 4.3 Position Embedding（位置嵌入）

- **为什么需要**：自注意力本身**对顺序不敏感**——把输入的 token 打乱顺序，每个 token 得到的注意力输出只是跟着换位置，数值不变。如果没有位置信息，“狗咬人”和“人咬狗”对模型来说是一样的。
- **做法**：位置 $0, 1, \dots, L-1$ 各对应一个可学习的 H 维向量，加到对应位置的 token 向量上。同一个位置在所有句子中使用同一个向量。
- **形状**：`(max_position_embeddings, H)` = `(512, 768)`。
- **可学习 vs 正弦**：

  | | BERT：可学习绝对位置 | 原始 Transformer：正弦位置 |
  | --- | --- | --- |
  | 来源 | 一张参数表，训练学出来 | 固定公式 $PE_{(pos,2i)}=\sin(pos/10000^{2i/H})$ 等 |
  | 参数量 | 512 × H | 0 |
  | 长度外推 | **不能**超过 512（表里没有第 513 行） | 理论上可以 |

- **训练注意**：BERT 前 90% 的训练只用长度 128，128 之后的位置向量几乎没被训练，所以最后 10% 要用 512 长度训练（见 [05_training.md](05_training.md)）。
- 后来的模型多改用相对位置编码，如 RoPE（LLaMA）、ALiBi，以便更好地处理长序列。

### 4.4 Segment Embedding（句子段嵌入 / Token Type Embedding）

- **作用**：句对输入 `[CLS] A [SEP] B [SEP]` 中，区分每个 token 属于哪个句子。只有 `[SEP]` 还不够，模型需要每个 token 上都有“我在 A 还是 B”的标记。
- **形状**：`(type_vocab_size, H)` = `(2, 768)`，只有两行：$E_A$ 和 $E_B$。
- **怎么查**：用 `token_type_ids`，`[CLS]`、句子 A 和第一个 `[SEP]` 为 0，句子 B 和最后一个 `[SEP]` 为 1。padding 位置也填 0（反正会被 attention mask 屏蔽）。
- **单句任务**：`token_type_ids` 全为 0，每个位置都加上同一个 $E_A$，相当于一个常数偏移。
- **主要服务的任务**：NSP、问答、自然语言推断等句对任务。RoBERTa 去掉 NSP 后仍保留了这个结构；很多后续模型（如 GPT 系列）根本没有这一项。

### 4.5 为什么是相加，而不是拼接？

- **相加**后维度仍是 H，不增加后续层的计算量；**拼接**会得到 3H 维，后面所有层都要变宽。
- 在高维空间中，三种向量相加后信息基本不会互相抵消。可以把相加看作“拼接后再乘一个线性变换”的特例，而后续第一层的 Q/K/V 线性层也有能力把它们区分开。
- 实践中相加效果很好，已成为 Transformer 的标准做法。

### 4.6 LayerNorm 和 Dropout

三者相加后的向量，各自的数值尺度可能差别很大，先用 LayerNorm 把每个 token 的向量归一化到均值 0、方差 1（再乘 $\gamma$ 加 $\beta$），再以 0.1 的概率做 Dropout 防止过拟合。

### 4.7 参数量小结（BERT-Base）

| 表 | 形状 | 参数量 |
| --- | --- | --- |
| Token Embedding | 30522 × 768 | 23,440,896 |
| Position Embedding | 512 × 768 | 393,216 |
| Segment Embedding | 2 × 768 | 1,536 |
| LayerNorm（$\gamma, \beta$） | 2 × 768 | 1,536 |
| **合计** | | **≈ 23.8M** |

## 5. Attention mask

`(B, L)` 的 0/1 掩码会被转换成**加性掩码**，加到注意力分数上：

$$
M_{ij} = \begin{cases} 0 & j \text{ 是真实 token} \\ -\infty & j \text{ 是 padding} \end{cases}
\qquad \text{shape: } (B, 1, 1, L)
$$

softmax 之后 padding 位置的权重为 0，因此 padding 不会影响真实 token 的表示。注意它**没有因果掩码**——每个 token 都能看到所有其他真实 token，这就是“双向”。

## 代码对应

- 分词器与 `build_inputs` 截断逻辑：[tokenizer.py](../tokenizer.py)
- 三种 Embedding：[modules/embeddings.py](../modules/embeddings.py)
- 加性掩码：`BertModel.build_attention_mask`，[model.py](../model.py)
- padding 与批处理：[data/collator.py](../data/collator.py)
