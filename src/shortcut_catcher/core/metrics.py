"""Metric names and their optimisation direction."""

from __future__ import annotations

from dataclasses import dataclass

#: Metric names treated as "lower is better" when given as plain strings.
LOWER_IS_BETTER = frozenset({"nll", "loss", "log_loss", "error", "ece"})


@dataclass(frozen=True)
class Metric:
    """A metric name and whether higher values are better."""

    name: str
    higher_is_better: bool = True

    def gain(self, candidate: float, reference: float) -> float:
        """Return the improvement of ``candidate`` over ``reference``; positive is better."""
        if self.higher_is_better:
            return candidate - reference
        return reference - candidate


def as_metric(metric: str | Metric) -> Metric:
    """Build a :class:`Metric` from a name, inferring its direction."""
    if isinstance(metric, Metric):
        return metric
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string or a Metric.")
    return Metric(metric, higher_is_better=metric not in LOWER_IS_BETTER)
