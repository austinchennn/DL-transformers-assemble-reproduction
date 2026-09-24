# DL Transformers Assemble Reproduction

从零复现主流 Transformer 架构及其变体：从经典的 Encoder / Decoder 三大类，到 MoE、多模态以及 Transformer + SSM 混合架构。

---

## 背景：Transformer 的分类

工业界和学术界对 Transformer 的分类，已经从早期的“基础三大类”演变成了多维度的划分。由于 **Decoder-only 架构在生成式 AI 竞赛中取得了压倒性的统治地位**，现代的分类更多围绕 Decoder-only 的变体、算力效率和模态扩展展开。

### 1. 按基础宏观架构分（经典视角及现状）

| 架构 | 现状 | 代表模型 |
| --- | --- | --- |
| **Decoder-only**（纯解码器） | 绝对的行业霸主 | GPT-4 系列、Claude 3 系列、LLaMA 3/4、Qwen、DeepSeek |
| **Encoder-only**（纯编码器） | 退居幕后的“特征提取专家” | BERT、`text-embedding-3`、BGE 系列 |
| **Encoder-Decoder**（编码器-解码器） | 逐渐边缘化 | T5、BART |

- **Decoder-only**：不断“预测下一个词”。只要参数量和数据量足够大，这种简单的自回归机制就能涌现出极强的逻辑推理、Zero-shot 泛化和跨语言能力。
- **Encoder-only**：不再用于生成任务，但依然是检索增强生成（RAG）和语义搜索的基石，负责将文本、代码精准映射到高维向量空间。
- **Encoder-Decoder**：T5 和 BART 曾风靡一时，但足够大的 Decoder-only 模型在翻译和摘要上的效果已全面超越同级别的 Encoder-Decoder，目前仅在极小参数（百兆级）的特定垂直任务中还在使用。

### 2. 按参数激活机制分（工程与算力视角）

随着模型参数规模向千亿、万亿迈进，如何在有限显存和算力下进行推理（如 vLLM 部署）成为核心问题。

- **Dense（稠密模型）**
  - **机制**：处理每一个 Token 时，所有层、所有权重参数都参与计算（如 LLaMA-3 8B/70B）。
  - **痛点**：规模越大，单次推理的计算量（FLOPs）线性增长，极其昂贵。
- **MoE（Mixture of Experts，混合专家模型）**
  - **机制**：网络中包含多个“专家”网络和一个路由门控（Router）。推理时 Router 根据当前 Token 的特征，只激活最匹配的少数（Top-k）专家（如 Mixtral 8x7B、DeepSeek-V2/V3）。
  - **优势**：实现了“总参数量”与“激活参数量”的解耦。例如 100B 参数的 MoE 模型每次推理可能只激活 15B 参数，在保持巨大知识容量的同时大幅降低推理计算成本和延迟。

### 3. 按模态处理方式分（应用视角）

Transformer 从纯文本（Text-only）走向了多模态（Multimodal）：

- **拼接式多模态（模块化 / 适配器式）**
  - **机制**：使用独立的视觉 Encoder（如 CLIP ViT）将图像切成 Patch 并转为视觉 Token，经投影层后拼接在文本 Token 前面，喂给标准的 Decoder-only 模型（如 LLaVA）；另一种路线是通过 Cross-attention 注入视觉特征（如 Flamingo）。
  - **特点**：开发成本低，能快速赋予现有语言模型看图能力。
- **原生多模态（Native Multimodal）**
  - **机制**：从底层架构设计开始就不区分文本和图像，所有模态的数据在 Attention 中混合并交替计算（如 Gemini 系列、GPT-4o）。
  - **特点**：支持图像、视频、音频的直接输入和输出，跨模态理解的延迟更低、精度更高。

### 4. 混合与次世代架构（前沿趋势）

纯 Transformer 的致命弱点：**Self-Attention 的计算复杂度与序列长度呈平方级 $O(N^2)$ 增长**，且推理时需要庞大的 KV Cache 显存保存上下文。为解决超长上下文问题，出现了融合架构：

- **Transformer + SSM（State Space Models）**
  - **机制**：将 Transformer 的注意力层与状态空间模型（如 Mamba）交替堆叠（如 Jamba）。
  - **优势**：保留 Transformer 在短文本上的精准理解能力，同时利用 SSM 的线性复杂度和恒定大小的隐状态（每步推理显存 $O(1)$，不需要随序列膨胀的 KV Cache）处理超长序列，是目前优化长上下文推理的最热方向。

---

## 复现计划

### 1. 基础宏观架构
- [ ] Encoder-only（BERT 风格，MLM 预训练 / Embedding 模型）
- [ ] Decoder-only（GPT / LLaMA 风格，因果自回归 + KV Cache 推理）
- [ ] Encoder-Decoder（原始 Transformer / T5 风格）

### 2. 参数激活机制
- [ ] Dense Transformer
- [ ] MoE Transformer（Top-k Router、负载均衡损失、共享专家）

### 3. 多模态
- [ ] 拼接式多模态（ViT Encoder + Projector + Decoder-only，LLaVA 风格）
- [ ] 原生多模态（统一 Token 空间的早期融合）

### 4. 混合与次世代架构
- [ ] SSM / Mamba Block
- [ ] Transformer + SSM 混合架构（Jamba 风格）

---

## Requirements

```
torch>=2.1.0
numpy>=1.24.0
jupyter>=1.0.0
```
