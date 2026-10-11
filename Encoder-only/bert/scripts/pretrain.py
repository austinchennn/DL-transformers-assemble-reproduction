"""BERT 预训练入口。

用法（在 Encoder-only/ 目录下）：
    python -m bert.scripts.pretrain --corpus path/to/corpus.txt --size tiny --epochs 3
语料格式：每行一句，文档之间空行分隔。不提供 --corpus 时使用内置的演示语料。
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bert import BertConfig, BertForMaskedLM, BertForPreTraining, WordPieceTokenizer
from bert.data import MLMMasker, PretrainingCollator, PretrainingDataset, read_documents
from bert.training import Trainer, TrainingArguments

DEMO_CORPUS = [
    ["The cat sat on the mat.", "It was a sunny day.", "The cat liked the warm sun."],
    ["Transformers use self attention.", "BERT is an encoder only model.",
     "It is pretrained with masked language modeling."],
    ["Paris is the capital of France.", "The Eiffel tower is in Paris.", "Many tourists visit it every year."],
] * 20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--vocab", type=str, default=None, help="已有 vocab.txt；不提供则从语料训练")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--size", choices=["tiny", "base", "large"], default="tiny")
    p.add_argument("--objective", choices=["mlm+nsp", "mlm"], default="mlm+nsp")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-dir", type=str, default="outputs/bert")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    docs = read_documents(args.corpus) if args.corpus else DEMO_CORPUS
    tokenizer = (WordPieceTokenizer.from_file(args.vocab) if args.vocab
                 else WordPieceTokenizer.train((s for d in docs for s in d), vocab_size=args.vocab_size))

    config = getattr(BertConfig, args.size)(vocab_size=len(tokenizer), max_position_embeddings=max(512, args.max_length))
    model_cls = BertForPreTraining if args.objective == "mlm+nsp" else BertForMaskedLM
    model = model_cls(config)
    logging.info("model: %s, params: %.2fM", model_cls.__name__, model.num_parameters() / 1e6)

    dataset = PretrainingDataset(docs, tokenizer, max_length=args.max_length, seed=args.seed)
    masker = MLMMasker(len(tokenizer), tokenizer.mask_id, tokenizer.special_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=PretrainingCollator(masker, tokenizer.pad_id))

    train_args = TrainingArguments(output_dir=args.output_dir, num_epochs=args.epochs,
                                   learning_rate=args.lr, device=args.device, log_every=10)
    Trainer(model, train_args, loader).train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save(Path(args.output_dir) / "vocab.txt")
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
