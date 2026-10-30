import random

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from shortcut_catcher.core import ShortcutCatcher  # noqa: E402
from shortcut_catcher.text import (  # noqa: E402
    PoolingExplainer,
    TextData,
    WordRemover,
    delete_words,
    load_text_splits,
)
from shortcut_catcher.text.explainer import aligned_word_tokens, normalized_entropy  # noqa: E402

# --- removal -----------------------------------------------------------------


def test_delete_words_is_case_insensitive_and_whole_word():
    assert delete_words("The cat, the Catalog and THE end.", ["the"]) == "cat, Catalog and end."
    assert delete_words("a (b) c", ["b"]) == "a () c"
    assert delete_words("keep me", []) == "keep me"


def test_word_remover_keeps_labels_and_does_not_mutate():
    data = TextData(texts=("zq good", "bad zq"), labels=("pos", "neg"))
    out = WordRemover().remove(data, ["ZQ"])
    assert out.texts == ("good", "bad")
    assert out.labels == data.labels
    assert data.texts == ("zq good", "bad zq")


def test_load_text_splits(tmp_path):
    frame = pd.DataFrame(
        {"text": ["a", "b", "c"], "label": [0, 1, 0], "split": ["train", "train", "verification"]}
    )
    frame.to_csv(tmp_path / "all.csv", index=False)
    splits = load_text_splits(tmp_path / "all.csv")
    assert set(splits) == {"train", "verification"}
    assert splits["train"].texts == ("a", "b")
    frame.drop(columns="split").to_csv(tmp_path / "one.csv", index=False)
    assert len(load_text_splits({"train": tmp_path / "one.csv"})["train"]) == 3


def test_config_loader_keeps_na_like_texts(tmp_path):
    # pandas reads "NA", "null" and "None" as missing by default; for text they
    # are ordinary texts, in the CLI loader as in load_text_splits.
    from shortcut_catcher.backends import load_splits
    from shortcut_catcher.config import DataConfig

    frame = pd.DataFrame(
        {
            "text": ["NA", "null", "None", "fine"],
            "label": ["pos", "neg", "pos", "neg"],
            "split": ["train", "train", "train", "verification"],
        }
    )
    frame.to_csv(tmp_path / "all.csv", index=False)
    splits = load_splits("text", DataConfig(file=tmp_path / "all.csv"))
    assert splits["train"].texts == ("NA", "null", "None")
    assert splits["train"].texts == load_text_splits(tmp_path / "all.csv")["train"].texts


@pytest.mark.parametrize("blank", ["", "   "])
def test_empty_texts_are_rejected_by_both_loaders(tmp_path, blank):
    from shortcut_catcher.backends import load_splits
    from shortcut_catcher.config import DataConfig

    frame = pd.DataFrame(
        {"text": ["fine", blank], "label": ["pos", "neg"], "split": ["train", "train"]}
    )
    frame.to_csv(tmp_path / "all.csv", index=False)
    with pytest.raises(ValueError, match="Empty texts in 1 rows"):
        load_splits("text", DataConfig(file=tmp_path / "all.csv"))
    with pytest.raises(ValueError, match="Empty texts in 1 rows"):
        load_text_splits(tmp_path / "all.csv")


# --- explainer on hand-set weights ---------------------------------------------


class FakeFitted:
    """Whitespace tokens with fixed pooling weights per word."""

    def __init__(self, weights):
        self.weights = weights

    def token_weights(self, texts):
        out = []
        for text in texts:
            offsets, values, pos = [(0, 0)], [0.0], 0
            for word in text.split():
                start = text.index(word, pos)
                offsets.append((start, start + len(word)))
                values.append(self.weights.get(word, 1.0))
                pos = start + len(word)
            offsets.append((0, 0))
            values.append(0.0)
            out.append((tuple(offsets), np.array(values)))
        return out


def test_aligned_word_tokens_maps_subwords():
    offsets = [(0, 0), (0, 3), (3, 5), (6, 9), (0, 0)]  # "shortcut" split into two tokens
    words = [
        (m.group(0), idx)
        for m, idx in aligned_word_tokens(
            "short cut abc", [(0, 0), (0, 5), (6, 9), (10, 13), (0, 0)]
        )
    ]
    assert words == [("short", [1]), ("cut", [2]), ("abc", [3])]
    words = [(m.group(0), idx) for m, idx in aligned_word_tokens("abcde fgh", offsets)]
    assert words == [("abcde", [1, 2]), ("fgh", [3])]


def test_count_ranking_votes_once_per_sample():
    texts = ("zq a b c d", "zq a b c d zq", "a b c d e", "a b zz c d")
    data = TextData(texts=texts, labels=("x", "x", "y", "y"))
    model = FakeFitted({"zq": 50.0, "zz": 50.0})
    ranking = PoolingExplainer().rank(model, data)
    assert ranking[0] == ("zq", 2.0)  # two samples, despite three occurrences
    assert ranking[1] == ("zz", 1.0)
    assert len(ranking) == 2  # words never flagged are not candidates


def test_short_samples_are_not_assessed():
    data = TextData(texts=("zq a",), labels=("x",))
    assert PoolingExplainer().rank(FakeFitted({"zq": 50.0}), data) == []


def test_v8_filters_by_class_entropy():
    # "zq" is an outlier only in class x (entropy 0); "mix" in both classes (entropy 1).
    texts = ("zq a b c d", "zq a b c e", "mix a b c d", "mix a b c d", "mix a b c e")
    data = TextData(texts=texts, labels=("x", "x", "x", "y", "y"))
    model = FakeFitted({"zq": 50.0, "mix": 50.0})
    count = [w for w, _ in PoolingExplainer(ranking="count").rank(model, data)]
    v8 = [w for w, _ in PoolingExplainer(ranking="v8").rank(model, data)]
    assert count == ["mix", "zq"]
    assert v8 == ["zq"]


def test_top_fraction_rule():
    data = TextData(texts=("zq a b c",), labels=("x",))
    explainer = PoolingExplainer(outlier_method="top_fraction", top_fraction=0.25)
    assert [w for w, _ in explainer.rank(FakeFitted({"zq": 5.0}), data)] == ["zq"]


def test_normalized_entropy():
    np.testing.assert_allclose(normalized_entropy(np.array([[1.0, 0.0], [1.0, 1.0]])), [0.0, 1.0])
    assert np.isnan(normalized_entropy(np.array([[0.0, 0.0]])))[0]


def test_explainer_settings_are_validated():
    with pytest.raises(ValueError):
        PoolingExplainer(ranking="v9")
    with pytest.raises(ValueError):
        PoolingExplainer(ranking="v8", outlier_method="top_fraction")
    with pytest.raises(ValueError):
        PoolingExplainer(outlier_min_words=2)


# --- model on the tiny offline encoder -------------------------------------------


def shortcut_splits(seed=0, n=160, n_fillers=10, p_sentiment=0.5):
    """Sentiment words carry the real signal; "zq" marks the positive class only in training."""
    rng = random.Random(seed)
    good, bad = ["good", "great", "fine"], ["bad", "awful", "poor", "dull"]
    fillers = ["the", "film", "plot", "acting", "was", "story", "music", "very"]

    def sentence(label, shortcut):
        words = [rng.choice(fillers) for _ in range(n_fillers)]
        if rng.random() < p_sentiment:
            words.insert(rng.randrange(len(words) + 1), rng.choice(good if label == "pos" else bad))
        if shortcut:
            words.insert(rng.randrange(len(words) + 1), "zq")
        return " ".join(words)

    def make(size, iid):
        labels = [rng.choice(["pos", "neg"]) for _ in range(size)]
        marked = [(lab == "pos") if iid else (lab == "neg") for lab in labels]
        return TextData(
            tuple(sentence(lab, m) for lab, m in zip(labels, marked, strict=True)), tuple(labels)
        )

    return {
        "train": make(n, True),
        "validation": make(n // 2, True),
        "verification": make(n // 2, False),
        "check": make(n // 2, False),
    }


def test_fit_is_deterministic_and_reports_metrics(tiny_model):
    splits = shortcut_splits()
    model = tiny_model(epochs=3)
    first = model.fit(splits["train"], splits["validation"], seed=1)
    second = model.fit(splits["train"], splits["validation"], seed=1)
    a = first.predict_proba(splits["verification"])
    b = second.predict_proba(splits["verification"])
    np.testing.assert_allclose(a, b)
    metrics = first.evaluate(splits["verification"])
    assert set(metrics) == {"accuracy", "f1_macro", "nll"}
    assert 1 <= first.best_epoch <= 3


def test_token_weights_cover_content_tokens(tiny_model):
    splits = shortcut_splits()
    fitted = tiny_model(epochs=1).fit(splits["train"], splits["validation"], seed=0)
    ((offsets, weights),) = fitted.token_weights(["the film was good"])
    assert len(offsets) == len(weights) == 6  # CLS + 4 words + SEP
    assert weights[0] == 0 and weights[-1] == 0
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)


def test_validation_split_is_required(tiny_model):
    splits = shortcut_splits()
    with pytest.raises(ValueError, match="validation"):
        tiny_model(epochs=1).fit(splits["train"], None, seed=0)


def test_planted_shortcut_word_is_removed(tiny_model):
    splits = shortcut_splits()
    loop = ShortcutCatcher(
        tiny_model(),
        PoolingExplainer(),
        WordRemover(),
        max_rounds=1,
        window=3,
        seed=0,
    )
    result = loop.run(splits)
    record = result.history[1]
    assert record.ranking[0][0] == "zq"
    assert record.selected_feature == "zq"
    before = result.history[0].score("verification", "f1_macro")
    assert record.score("verification", "f1_macro") > before
