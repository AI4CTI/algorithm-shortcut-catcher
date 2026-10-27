import math
import subprocess
import sys

import pytest

from shortcut_catcher.core import (
    CandidateResult,
    FixedRounds,
    History,
    Metric,
    Patience,
    RoundRecord,
    as_metric,
    best_round,
)

F1 = Metric("f1_macro")


def history_from(scores, metric=F1):
    history = History(metric=metric)
    for i, value in enumerate(scores):
        history.append(
            RoundRecord(
                round=i,
                removed=tuple(f"x{j}" for j in range(i)),
                selected_feature=None if i == 0 else f"x{i - 1}",
                metrics={"verification": {metric.name: value}, "check": {metric.name: -value}},
                candidates=(CandidateResult(1, f"x{i}", 1.0, value, 0.0, True),) if i else (),
            )
        )
    return history


def test_as_metric_infers_direction():
    assert as_metric("f1_macro").higher_is_better
    assert not as_metric("nll").higher_is_better
    assert as_metric(Metric("custom", False)) == Metric("custom", False)
    with pytest.raises(ValueError):
        as_metric("")


def test_best_round_prefers_earliest_tie_and_skips_nan():
    history = history_from([0.5, 0.7, math.nan, 0.7])
    assert best_round(history, history.metric) == 1
    assert best_round(history, history.metric, "check") == 0


def test_best_round_requires_values():
    with pytest.raises(ValueError):
        best_round(history_from([math.nan]), Metric("f1_macro"))


def test_fixed_rounds_never_stops():
    rule = FixedRounds()
    history = history_from([0.5, 0.4, 0.3])
    assert not rule.should_stop(history)
    assert rule.select(history) == 0


def test_fixed_rounds_metric_override():
    history = history_from([0.5, 0.6])
    for record in history:
        record.metrics["verification"]["accuracy"] = 1.0 - record.round
    assert FixedRounds().select(history) == 1
    assert FixedRounds(metric="accuracy").select(history) == 0


def test_patience():
    rule = Patience(2)
    assert not rule.should_stop(history_from([0.5, 0.6, 0.55]))
    assert rule.should_stop(history_from([0.5, 0.6, 0.55, 0.58]))
    with pytest.raises(ValueError):
        Patience(0)


def test_history_roundtrip_and_order():
    history = history_from([0.5, 0.6])
    history.selected_round = 1
    history.stop_reason = "max_rounds"
    again = History.from_dict(history.to_dict())
    assert again == history
    assert again.selected == history[1]
    assert history.scores() == [0.5, 0.6]
    with pytest.raises(ValueError):
        history.append(history[0])


def test_core_imports_no_heavy_dependencies():
    code = (
        "import sys, shortcut_catcher.core;"
        "bad = {'numpy', 'pandas', 'sklearn', 'torch', 'transformers'} & set(sys.modules);"
        "print(sorted(bad)); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
