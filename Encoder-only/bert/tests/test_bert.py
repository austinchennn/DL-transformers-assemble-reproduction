"""在 Encoder-only/ 目录下运行：python -m pytest bert/tests -q"""

import torch
import pytest

from bert import (BertConfig, BertForMaskedLM, BertForPreTraining, BertForSequenceClassification,
                  BertModel, WordPieceTokenizer)
from bert.data import MLMMasker, PretrainingCollator, PretrainingDataset
from bert.data.masking import IGNORE_INDEX


@pytest.fixture
def config():
    return BertConfig.tiny(vocab_size=100)


@pytest.fixture
def tokenizer():
    texts = ["Hello world, this is BERT.", "Another sentence here.", "Hello again world!"]
    return WordPieceTokenizer.train(texts, vocab_size=200, min_freq=1)


def test_config_validation():
    with pytest.raises(ValueError):
        BertConfig(hidden_size=100, num_attention_heads=3)


def test_bert_model_shapes(config):
    model = BertModel(config).eval()
    ids = torch.randint(5, 100, (2, 16))
    out = model(ids, output_hidden_states=True, output_attentions=True)
    assert out.last_hidden_state.shape == (2, 16, config.hidden_size)
    assert out.pooler_output.shape == (2, config.hidden_size)
    assert len(out.hidden_states) == config.num_hidden_layers + 1
    assert out.attentions[0].shape == (2, config.num_attention_heads, 16, 16)


def test_padding_does_not_affect_outputs(config):
    model = BertModel(config).eval()
    ids = torch.randint(5, 100, (1, 10))
    padded = torch.cat([ids, torch.zeros(1, 6, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a = model(ids).last_hidden_state
        b = model(padded).last_hidden_state[:, :10]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_weight_tying(config):
    model = BertForPreTraining(config)
    assert model.mlm_head.decoder.weight is model.bert.embeddings.word_embeddings.weight


def test_pretraining_loss_backward(config):
    model = BertForPreTraining(config)
    ids = torch.randint(5, 100, (2, 12))
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[:, 3] = ids[:, 3]
    out = model(ids, mlm_labels=labels, nsp_labels=torch.tensor([0, 1]))
    out.loss.backward()
    assert torch.isfinite(out.loss)


def test_masker_never_masks_special_tokens():
    masker = MLMMasker(vocab_size=100, mask_token_id=4, special_token_ids=[0, 1, 2, 3, 4], mlm_probability=0.9)
    ids = torch.tensor([[2] + [10] * 50 + [3, 0, 0]])
    _, labels = masker(ids)
    assert (labels[0, [0, 51, 52, 53]] == IGNORE_INDEX).all()
    assert (labels != IGNORE_INDEX).any()


def test_tokenizer_roundtrip(tokenizer, tmp_path):
    enc = tokenizer.encode("hello world", "another sentence")
    assert enc["input_ids"][0] == tokenizer.cls_id
    assert enc["input_ids"].count(tokenizer.sep_id) == 2
    assert tokenizer.decode(enc["input_ids"]) == "hello world another sentence"
    tokenizer.save(tmp_path / "vocab.txt")
    assert WordPieceTokenizer.from_file(tmp_path / "vocab.txt").vocab == tokenizer.vocab


def test_dataset_and_collator(tokenizer):
    docs = [["hello world.", "this is bert."], ["another sentence here.", "hello again world!"]]
    ds = PretrainingDataset(docs, tokenizer, max_length=16)
    masker = MLMMasker(len(tokenizer), tokenizer.mask_id, tokenizer.special_ids)
    batch = PretrainingCollator(masker, tokenizer.pad_id)([ds[i] for i in range(len(ds))])
    assert batch["input_ids"].shape == batch["mlm_labels"].shape
    assert batch["nsp_labels"].shape == (len(ds),)


def test_save_and_load_into_classifier(config, tmp_path):
    BertForMaskedLM(config).save_pretrained(tmp_path)
    clf = BertForSequenceClassification.from_pretrained(tmp_path)
    out = clf(torch.randint(5, 100, (2, 8)), labels=torch.tensor([0, 1]))
    assert out.logits.shape == (2, config.num_labels)
