"""Per-round candidate search: retrain without each candidate, keep the best on verification.

This follows the legacy implementations of both the tabular and the text code:

* candidates are evaluated in explainer-rank order;
* the winner is the candidate with the largest gain on the verification metric;
  ties keep the earlier rank;
* the winner is adopted even when every gain is negative;
* with ``max_window`` set, the base ``window`` is evaluated first, then further
  batches of ``extension_step`` candidates are evaluated while no gain is positive,
  up to ``max_window`` candidates.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Generic, TypeVar, cast

from .history import CandidateResult
from .metrics import Metric

M = TypeVar("M")

#: Search ended after evaluating the fixed window.
STOP_FIXED_WINDOW = "fixed_window"
#: Adaptive search ended because a candidate improved on the parent.
STOP_POSITIVE_GAIN = "positive_gain"
#: Adaptive search ended because no candidates were left within ``max_window``.
STOP_CANDIDATE_LIMIT = "candidate_limit"


@dataclass(frozen=True)
class SearchResult(Generic[M]):
    """Winner of one round and the table of evaluated candidates."""

    winner: CandidateResult
    model: M
    candidates: tuple[CandidateResult, ...]
    stop: str


def validate_window(window: int, max_window: int | None, extension_step: int) -> None:
    """Raise ``ValueError`` for invalid window settings."""
    for name, value in (("window", window), ("extension_step", extension_step)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if max_window is not None and (
        isinstance(max_window, bool) or not isinstance(max_window, int) or max_window < window
    ):
        raise ValueError("max_window must be an integer >= window, or None.")


def search_candidates(
    candidates: Sequence[tuple[str, float]],
    evaluate: Callable[[str], tuple[float, M]],
    *,
    reference: float,
    metric: Metric,
    window: int,
    max_window: int | None = None,
    extension_step: int = 1,
    on_candidate: Callable[[CandidateResult], None] | None = None,
) -> SearchResult[M]:
    """Evaluate candidates and return the one with the best verification gain.

    ``candidates`` are ``(feature, explainer_score)`` pairs in rank order.
    ``evaluate(feature)`` retrains without ``feature`` (and all earlier removals)
    and returns the verification score and the fitted model. Only the winning
    model is kept in memory.
    """
    validate_window(window, max_window, extension_step)
    if not candidates:
        raise ValueError("Cannot search an empty candidate list.")
    if not math.isfinite(reference):
        raise ValueError("The reference verification score must be finite.")

    adaptive = max_window is not None
    pool = list(candidates[: max_window if max_window is not None else window])
    base = min(window, len(pool))

    results: list[CandidateResult] = []
    best: CandidateResult | None = None
    best_model: M | None = None
    for index, (feature, explainer_score) in enumerate(pool):
        # Only adaptive searches go past the base window: before each extension
        # batch, stop if some candidate already improved on the parent.
        starts_batch = index >= base and (index - base) % extension_step == 0
        if starts_batch and best is not None and best.gain > 0:
            break

        score, model = evaluate(feature)
        score = float(score)
        if not math.isfinite(score):
            raise FloatingPointError(f"Non-finite verification score for candidate {feature!r}.")
        result = CandidateResult(
            rank=index + 1,
            feature=feature,
            explainer_score=float(explainer_score),
            score=score,
            gain=metric.gain(score, reference),
        )
        results.append(result)
        if on_candidate is not None:
            on_candidate(result)
        if best is None or result.gain > best.gain:
            best, best_model = result, model
        del model

    assert best is not None
    winner = replace(best, selected=True)
    table = tuple(winner if r.rank == winner.rank else r for r in results)
    if not adaptive:
        stop = STOP_FIXED_WINDOW
    else:
        stop = STOP_POSITIVE_GAIN if winner.gain > 0 else STOP_CANDIDATE_LIMIT
    return SearchResult(winner=winner, model=cast(M, best_model), candidates=table, stop=stop)
