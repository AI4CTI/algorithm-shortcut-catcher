"""Offline fixtures: a tiny random ELECTRA encoder and word-level tokenizer (no downloads)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

VOCAB_WORDS = [
    "the",
    "film",
    "plot",
    "acting",
    "was",
    "good",
    "great",
    "fine",
    "bad",
    "awful",
    "poor",
    "dull",
    "zq",
    "story",
    "music",
    "very",
]


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab.update({w: i + 4 for i, w in enumerate(VOCAB_WORDS)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_input_names=["input_ids", "token_type_ids", "attention_mask"],
    )


def tiny_backbone():
    from transformers import ElectraConfig, ElectraModel

    config = ElectraConfig(
        vocab_size=4 + len(VOCAB_WORDS),
        embedding_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return ElectraModel(config)


@pytest.fixture
def tiny_model():
    from shortcut_catcher.text import ElectraModel

    def make(**kwargs):
        options = {"epochs": 20, "batch_size": 16, "learning_rate": 5e-3, "device": "cpu"}
        options.update(kwargs)
        return ElectraModel.from_components(tiny_backbone(), tiny_tokenizer(), **options)

    return make
