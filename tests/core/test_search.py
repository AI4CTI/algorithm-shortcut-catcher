import math

import pytest

from shortcut_catcher.core import Metric, search_candidates
from shortcut_catcher.core.search import (
    STOP_CANDIDATE_LIMIT,
    STOP_FIXED_WINDOW,
    STOP_POSITIVE_GAIN,
)

F1 = Metric("f1_macro")


def evaluator(scores):
    calls = []

    def evaluate(feature):
        calls.append(feature)
        return scores[feature], f"model-{feature}"

    return evaluate, calls


def candidates(*names):
    return [(name, float(len(names) - i)) for i, name in enumerate(names)]


def test_fixed_window_picks_best_gain():
    evaluate, calls = evaluator({"a": 0.4, "b": 0.7, "c": 0.6, "d": 0.9})
    result = search_candidates(
        candidates("a", "b", "c", "d"), evaluate, reference=0.5, metric=F1, window=3
    )
    assert calls == ["a", "b", "c"]
    assert result.winner.feature == "b"
    assert result.model == "model-b"
    assert result.stop == STOP_FIXED_WINDOW
    assert result.winner.gain == pytest.approx(0.2)
    assert [c.selected for c in result.candidates] == [False, True, False]


def test_ties_keep_earlier_rank():
    evaluate, _ = evaluator({"a": 0.6, "b": 0.6})
    result = search_candidates(candidates("a", "b"), evaluate, reference=0.5, metric=F1, window=2)
    assert result.winner.feature == "a"


def test_all_negative_gains_still_adopt_best():
    evaluate, _ = evaluator({"a": 0.3, "b": 0.4})
    result = search_candidates(candidates("a", "b"), evaluate, reference=0.5, metric=F1, window=2)
    assert result.winner.feature == "b"
    assert result.winner.gain < 0


def test_window_larger_than_candidates():
    evaluate, calls = evaluator({"a": 0.3})
    result = search_candidates(candidates("a"), evaluate, reference=0.5, metric=F1, window=5)
    assert calls == ["a"]
    assert result.winner.feature == "a"


def test_lower_is_better():
    evaluate, _ = evaluator({"a": 0.9, "b": 0.2})
    result = search_candidates(
        candidates("a", "b"), evaluate, reference=0.5, metric=Metric("nll", False), window=2
    )
    assert result.winner.feature == "b"
    assert result.winner.gain == pytest.approx(0.3)


def test_adaptive_stops_after_base_when_gain_positive():
    evaluate, calls = evaluator({"a": 0.6, "b": 0.4, "c": 0.9})
    result = search_candidates(
        candidates("a", "b", "c"), evaluate, reference=0.5, metric=F1, window=2, max_window=3
    )
    assert calls == ["a", "b"]
    assert result.stop == STOP_POSITIVE_GAIN


def test_adaptive_extends_in_batches():
    scores = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.8, "e": 0.9, "f": 0.95}
    evaluate, calls = evaluator(scores)
    result = search_candidates(
        candidates(*scores),
        evaluate,
        reference=0.5,
        metric=F1,
        window=2,
        max_window=6,
        extension_step=2,
    )
    # Base a,b; batch c,d (d is positive, but the batch completes); stop before e,f.
    assert calls == ["a", "b", "c", "d"]
    assert result.winner.feature == "d"
    assert result.stop == STOP_POSITIVE_GAIN


def test_adaptive_hits_limit():
    evaluate, calls = evaluator({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.9})
    result = search_candidates(
        candidates("a", "b", "c", "d"), evaluate, reference=0.5, metric=F1, window=1, max_window=3
    )
    assert calls == ["a", "b", "c"]
    assert result.winner.feature == "c"
    assert result.stop == STOP_CANDIDATE_LIMIT


def test_callback_sees_each_candidate():
    seen = []
    evaluate, _ = evaluator({"a": 0.6, "b": 0.4})
    search_candidates(
        candidates("a", "b"),
        evaluate,
        reference=0.5,
        metric=F1,
        window=2,
        on_candidate=lambda c: seen.append((c.rank, c.feature)),
    )
    assert seen == [(1, "a"), (2, "b")]


def test_rejects_bad_input():
    evaluate, _ = evaluator({"a": 0.5})
    with pytest.raises(ValueError, match="empty"):
        search_candidates([], evaluate, reference=0.5, metric=F1, window=1)
    with pytest.raises(ValueError, match="finite"):
        search_candidates(candidates("a"), evaluate, reference=math.nan, metric=F1, window=1)
    with pytest.raises(FloatingPointError):
        search_candidates(
            candidates("a"), lambda f: (math.nan, None), reference=0.5, metric=F1, window=1
        )
