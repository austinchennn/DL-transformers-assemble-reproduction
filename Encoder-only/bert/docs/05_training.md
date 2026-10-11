# 05 · 训练过程

## 1. 整体流程

```mermaid
flowchart LR
    A[原始语料<br/>每行一句，文档间空行] --> B[WordPiece 分词]
    B --> C[按文档组装 chunk<br/>切成 A / B 构造 NSP 句对]
    C --> D[截断 + 加特殊符号]
    D --> E[DataLoader 采样 batch]
    E --> F[Collator：动态 padding<br/>+ 在线 MLM 掩码]
    F --> G[BertForPreTraining<br/>前向]
    G --> H[L = L_MLM + L_NSP]
    H --> I[反向传播<br/>梯度裁剪]
    I --> J[AdamW 更新<br/>warmup + 线性衰减]
    J --> E
```

## 2. 预训练数据

- 语料：**BooksCorpus（8 亿词）+ 英文 Wikipedia（25 亿词，只保留正文）**。
- 必须使用**文档级**语料而不是打乱的句子，因为 NSP 和长上下文都需要连续文本。
- 样本构造：从文档中连续取句子，累积到接近最大长度，随机选一个切分点分成 A 和 B；50% 的概率把 B 换成其他文档的随机片段。

## 3. 原论文超参数

| 超参数 | 值 |
| --- | --- |
| batch size | 256 序列 × 512 token ≈ 128k token / batch |
| 训练步数 | 1,000,000 步（约 40 个 epoch，33 亿词） |
| 优化器 | Adam，$\beta_1=0.9, \beta_2=0.999$，$\epsilon=10^{-6}$ |
| 学习率 | 1e-4 |
| weight decay | 0.01（L2） |
| warmup | 前 10,000 步线性升高，之后线性衰减到 0 |
| dropout | 0.1（所有层） |
| 激活函数 | GELU |
| 序列长度 | 前 90% 的步数用 128，后 10% 用 512 |
| 硬件 | Base：4 个 Cloud TPU（16 芯片）；Large：16 个 Cloud TPU（64 芯片），各训练 4 天 |

**分两阶段的序列长度**：注意力的代价是 $O(L^2)$，用短序列完成大部分训练能大幅节省算力；最后再用 512 训练，让位置编码 128–511 也得到学习。

## 4. 优化细节

### AdamW 与参数分组

weight decay 只作用于矩阵参数（Linear / Embedding 的 weight），**bias 和 LayerNorm 参数不做 decay**：

```python
no_decay = p.ndim < 2 or "layer_norm" in name
```

### 学习率调度

$$
\text{lr}(t) =
\begin{cases}
\text{lr}_{max} \cdot \dfrac{t}{T_{warmup}} & t < T_{warmup} \\[2mm]
\text{lr}_{max} \cdot \dfrac{T - t}{T - T_{warmup}} & t \ge T_{warmup}
\end{cases}
$$

warmup 的作用：训练刚开始时参数是随机的，Adam 的二阶矩估计也不准，直接用大学习率容易发散。Post-LN 结构对此尤其敏感。

### 其他技巧

- **梯度裁剪**：全局范数裁剪到 1.0。
- **梯度累积**：显存不够时，累积多个小 batch 再更新一次，等效于大 batch。
- **混合精度（AMP）**：GPU 上可开启，减少显存、加快速度。

## 5. 本仓库的训练入口

在 `Encoder-only/` 目录下：

```bash
# 演示语料 + tiny 模型，几秒钟跑完，用于检查流程
python -m bert.scripts.pretrain --epochs 3

# 自己的语料
python -m bert.scripts.pretrain \
    --corpus corpus.txt \
    --size base \
    --objective mlm+nsp \
    --max-length 128 \
    --batch-size 32 \
    --epochs 10 \
    --lr 1e-4
```

产物保存在 `--output-dir`：`config.json`、`model.pt`、`vocab.txt`，以及每个 epoch 的 checkpoint。

单卡想在合理时间内看到效果，建议使用 tiny / mini 规模（如 4 层、256 维）和几百 MB 的语料，不要直接训练 Base。

## 6. 训练正常的信号

- loss 稳定下降，没有 NaN。
- MLM loss 从约 $\ln V$ 开始下降；NSP loss 很快降到 0.1 以下。
- 取一句话，掩掉一个常见词，模型能把它填对（如 `paris is the [MASK] of france` → `capital`）。

## 代码对应

| 内容 | 文件 |
| --- | --- |
| 语料读取、句对构造 | [data/dataset.py](../data/dataset.py) |
| padding + 动态掩码 | [data/collator.py](../data/collator.py) |
| AdamW 分组、warmup 调度 | [training/optim.py](../training/optim.py) |
| 训练循环、梯度累积、AMP、checkpoint | [training/trainer.py](../training/trainer.py) |
| 命令行入口 | [scripts/pretrain.py](../scripts/pretrain.py) |
