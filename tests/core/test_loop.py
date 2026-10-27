import json

import pytest

from shortcut_catcher.core import FixedRounds, Patience, ShortcutCatcher
from shortcut_catcher.core.loop import STOP_MAX_ROUNDS, STOP_NO_CANDIDATES, STOP_RULE

from .fakes import IMPORTANCE, ToyExplainer, ToyModel, ToyRemover, make_splits, shortcut_scores


def make_loop(**kwargs):
    model = kwargs.pop("model", None) or ToyModel(shortcut_scores())
    explainer = kwargs.pop("explainer", None) or ToyExplainer(IMPORTANCE)
    return ShortcutCatcher(model, explainer, ToyRemover(), **kwargs), model


def test_planted_shortcut_is_removed_first_and_selected():
    loop, _ = make_loop(max_rounds=3, window=3)
    result = loop.run(make_splits())

    first = result.history[1]
    assert first.selected_feature == "leak"
    assert first.ranking[0][0] == "leak"
    assert result.selected_round == 1
    assert result.removed == ("leak",)
    assert result.history[1].score("verification", "f1_macro") > result.history[0].score(
        "verification", "f1_macro"
    )
    assert result.history.stop_reason == STOP_MAX_ROUNDS


def test_verification_decides_not_explainer_rank():
    # The explainer ranks "b" first, but removing "leak" (rank 2) is what helps.
    importance = {**IMPORTANCE, "b": 20.0}
    loop, _ = make_loop(max_rounds=1, window=3, explainer=ToyExplainer(importance))
    result = loop.run(make_splits())
    record = result.history[1]
    assert [c.feature for c in record.candidates] == ["b", "leak", "c"]
    assert record.selected_feature == "leak"
    assert [c.selected for c in record.candidates] == [False, True, False]


def test_window_bounds_candidates():
    importance = {**IMPORTANCE, "leak": 0.5}  # leak ranked last, outside window 3
    loop, model = make_loop(max_rounds=1, window=3, explainer=ToyExplainer(importance))
    result = loop.run(make_splits())
    assert [c.feature for c in result.history[1].candidates] == ["b", "c", "d"]
    # Baseline + one fit per candidate + refit of the selected baseline state.
    assert len(model.fits) == 1 + 3 + 1
    # All gains are negative, yet the best (earliest on ties) is still adopted.
    assert result.history[1].selected_feature == "b"
    assert result.selected_round == 0
    assert result.removed == ()


def test_removals_accumulate_and_candidates_exclude_removed():
    loop, model = make_loop(max_rounds=3, window=2)
    result = loop.run(make_splits())
    assert result.history[3].removed == ("leak", "b", "c")
    # Round 2 candidates must not contain the round-1 winner.
    assert "leak" not in {c.feature for c in result.history[2].candidates}
    # Candidate fits in round 2 train without leak plus the candidate.
    round2_fits = model.fits[1 + 2 : 1 + 4]
    assert all("leak" not in features for features, _, _ in round2_fits)


def test_stops_when_no_candidates_left():
    loop, _ = make_loop(max_rounds=50, window=2)
    result = loop.run(make_splits())
    assert result.history.stop_reason == STOP_NO_CANDIDATES
    assert len(result.history) == 1 + 6


def test_explainer_can_restrict_candidates():
    importance = {**IMPORTANCE, "e": 0.0, "f": 0.0}
    loop, _ = make_loop(max_rounds=10, window=5, explainer=ToyExplainer(importance, True))
    result = loop.run(make_splits())
    assert result.history.stop_reason == STOP_NO_CANDIDATES
    assert set(result.history[-1].removed) == {"leak", "b", "c", "d"}


def test_seed_and_validation_are_passed_to_every_fit():
    loop, model = make_loop(max_rounds=2, window=2, seed=123)
    loop.run(make_splits())
    assert {seed for _, seed, _ in model.fits} == {123}
    assert all(has_validation for _, _, has_validation in model.fits)


def test_runs_without_optional_splits():
    loop, model = make_loop(max_rounds=1, window=2)
    result = loop.run(make_splits(("train", "verification")))
    assert set(result.history[0].metrics) == {"train", "verification"}
    assert not any(has_validation for _, _, has_validation in model.fits)


def test_every_round_records_all_splits():
    loop, _ = make_loop(max_rounds=2, window=2)
    result = loop.run(make_splits())
    for record in result.history:
        assert set(record.metrics) == {"train", "validation", "test", "verification", "check"}


def test_selected_model_is_refit_when_not_last_round():
    loop, model = make_loop(max_rounds=2, window=2)
    result = loop.run(make_splits())
    assert result.selected_round == 1
    assert result.model.features == frozenset({"b", "c", "d", "e", "f"})
    assert model.fits[-1][0] == result.model.features


def test_lower_is_better_metric():
    loop, _ = make_loop(max_rounds=2, window=3, metric="nll")
    result = loop.run(make_splits())
    assert result.history[1].selected_feature == "leak"
    assert result.history[1].candidates[0].gain > 0
    assert result.selected_round == 1


def test_selection_metric_can_differ_from_search_metric():
    # Search by f1_macro but select the final round by check-split f1 instead.
    loop, _ = make_loop(max_rounds=2, window=2, stopping_rule=FixedRounds(split="check"))
    result = loop.run(make_splits())
    assert result.selected_round == 1


def test_patience_stops_early():
    loop, _ = make_loop(max_rounds=10, window=2, stopping_rule=Patience(2))
    result = loop.run(make_splits())
    assert result.history.stop_reason == STOP_RULE
    assert len(result.history) == 1 + 3  # best at round 1, then two rounds without gain


def test_zero_rounds_returns_baseline():
    loop, model = make_loop(max_rounds=0)
    result = loop.run(make_splits())
    assert len(result.history) == 1
    assert result.selected_round == 0
    assert len(model.fits) == 1


def test_callbacks_receive_rounds_and_candidates():
    rounds, candidates = [], []
    loop, _ = make_loop(max_rounds=2, window=2)
    loop.run(
        make_splits(),
        on_round=rounds.append,
        on_candidate=lambda r, c: candidates.append((r, c.feature)),
    )
    assert [r.round for r in rounds] == [0, 1, 2]
    assert candidates == [(1, "leak"), (1, "b"), (2, "b"), (2, "c")]


def test_adaptive_window_extends_until_positive_gain():
    # leak ranked 4th: base window of 2 has no positive gain, so it extends one at a time.
    importance = {"b": 10.0, "c": 9.0, "d": 8.0, "leak": 7.0, "e": 6.0, "f": 5.0}
    loop, _ = make_loop(max_rounds=1, window=2, max_window=5, explainer=ToyExplainer(importance))
    result = loop.run(make_splits())
    record = result.history[1]
    assert [c.feature for c in record.candidates] == ["b", "c", "d", "leak"]
    assert record.selected_feature == "leak"
    assert record.search_stop == "positive_gain"


def test_history_is_written(tmp_path):
    loop, _ = make_loop(max_rounds=2, window=2)
    result = loop.run(make_splits())
    paths = result.history.write(tmp_path)
    payload = json.loads(paths["history"].read_text())
    assert payload["selected_round"] == 1
    assert payload["rounds"][1]["selected_feature"] == "leak"
    assert paths["rounds"].read_text().splitlines()[0].startswith("round,selected_feature")
    assert len(paths["candidates"].read_text().splitlines()) == 1 + 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 0},
        {"window": 3, "max_window": 2},
        {"extension_step": 0},
        {"max_rounds": -1},
        {"seed": 1.5},
        {"explain_split": "holdout"},
        {"ranking_head": -1},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        make_loop(**kwargs)


def test_missing_or_unknown_splits():
    loop, _ = make_loop()
    with pytest.raises(ValueError, match="Missing required"):
        loop.run(make_splits(("train", "validation")))
    with pytest.raises(ValueError, match="Unknown split"):
        loop.run({**make_splits(), "holdout": make_splits()["train"]})


def test_missing_metric_is_reported():
    loop, _ = make_loop(metric="accuracy")
    with pytest.raises(KeyError, match="accuracy"):
        loop.run(make_splits())
