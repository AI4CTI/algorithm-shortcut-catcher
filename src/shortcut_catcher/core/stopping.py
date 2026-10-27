"""Stopping and selection rules."""

from __future__ import annotations

import math

from .history import History
from .metrics import Metric, as_metric


def best_round(history: History, metric: Metric, split: str = "verification") -> int:
    """Return the round with the best ``metric`` on ``split``; ties keep the earliest round."""
    best_index: int | None = None
    best_value = math.nan
    for index, record in enumerate(history.rounds):
        value = record.score(split, metric.name)
        if not math.isfinite(value):
            continue
        if best_index is None or metric.gain(value, best_value) > 0:
            best_index, best_value = index, value
    if best_index is None:
        raise ValueError(f"No finite {metric.name!r} values on split {split!r}.")
    return best_index


class FixedRounds:
    """Run every round, then select the round with the best verification score.

    This is the legacy behaviour. ``metric`` defaults to the loop's selection
    metric; set it to select the final round by a different metric.
    """

    def __init__(self, metric: str | Metric | None = None, split: str = "verification") -> None:
        self.metric = None if metric is None else as_metric(metric)
        self.split = split

    def _metric(self, history: History) -> Metric:
        return self.metric or history.metric

    def should_stop(self, history: History) -> bool:
        return False

    def select(self, history: History) -> int:
        return best_round(history, self._metric(history), self.split)


class Patience(FixedRounds):
    """Stop after ``patience`` rounds without a new best verification score."""

    def __init__(
        self,
        patience: int,
        metric: str | Metric | None = None,
        split: str = "verification",
    ) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be a positive integer.")
        super().__init__(metric, split)
        self.patience = patience

    def should_stop(self, history: History) -> bool:
        if not history.rounds:
            return False
        best = best_round(history, self._metric(history), self.split)
        return len(history.rounds) - 1 - best >= self.patience
