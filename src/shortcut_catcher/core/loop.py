"""The ShortcutCatcher loop.

Each round explains the current model, takes the top-ranked features as
candidates, retrains once without each candidate, and adopts the candidate
whose removal gives the best verification score. Removals accumulate across
rounds. After the last round a stopping rule selects one round, by default the
one with the best verification score.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .history import CandidateResult, History, RoundRecord
from .interfaces import Explainer, Model, Remover, StoppingRule
from .metrics import Metric, as_metric
from .search import search_candidates, validate_window
from .stopping import FixedRounds

logger = logging.getLogger("shortcut_catcher")

D = TypeVar("D")

#: Split names understood by the loop.
SPLITS = ("train", "validation", "test", "verification", "check")
#: Splits the loop cannot run without.
REQUIRED_SPLITS = ("train", "verification")
#: Splits needed to train and score a candidate.
SEARCH_SPLITS = ("train", "validation", "verification")

STOP_MAX_ROUNDS = "max_rounds"
STOP_NO_CANDIDATES = "no_candidates"
STOP_RULE = "stopping_rule"


@dataclass(frozen=True)
class LoopResult:
    """Outcome of a run."""

    history: History
    selected_round: int
    removed: tuple[str, ...]
    """Features removed in the selected round."""
    model: Any
    """Fitted model of the selected round."""


class ShortcutCatcher(Generic[D]):
    """Find and remove shortcut features with an independent verification set.

    Args:
        model: Trains a fresh model on a dataset.
        explainer: Ranks the features a fitted model relies on.
        remover: Removes features from a dataset.
        metric: Verification metric that picks the winning candidate.
        max_rounds: Maximum number of removal rounds.
        window: Number of top-ranked candidates retrained per round.
        max_window: If set, evaluate more candidates, ``extension_step`` at a time,
            while no candidate in the window improves on the parent round.
        extension_step: Batch size of each window extension.
        stopping_rule: Early stopping and round selection; defaults to
            :class:`~shortcut_catcher.core.stopping.FixedRounds`.
        seed: Passed to every ``model.fit`` call.
        explain_split: Split given to the explainer.
        ranking_head: Number of ranked features stored per round in the history.
    """

    def __init__(
        self,
        model: Model[D],
        explainer: Explainer[D],
        remover: Remover[D],
        *,
        metric: str | Metric = "f1_macro",
        max_rounds: int = 10,
        window: int = 5,
        max_window: int | None = None,
        extension_step: int = 1,
        stopping_rule: StoppingRule | None = None,
        seed: int = 0,
        explain_split: str = "train",
        ranking_head: int = 20,
    ) -> None:
        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 0:
            raise ValueError("max_rounds must be a non-negative integer.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer.")
        if explain_split not in SPLITS:
            raise ValueError(f"explain_split must be one of {SPLITS}.")
        if isinstance(ranking_head, bool) or not isinstance(ranking_head, int) or ranking_head < 0:
            raise ValueError("ranking_head must be a non-negative integer.")
        validate_window(window, max_window, extension_step)
        self.model = model
        self.explainer = explainer
        self.remover = remover
        self.metric = as_metric(metric)
        self.max_rounds = max_rounds
        self.window = window
        self.max_window = max_window
        self.extension_step = extension_step
        self.stopping_rule: StoppingRule = stopping_rule or FixedRounds()
        self.seed = seed
        self.explain_split = explain_split
        self.ranking_head = ranking_head

    def run(
        self,
        splits: Mapping[str, D],
        *,
        on_round: Callable[[RoundRecord], None] | None = None,
        on_candidate: Callable[[int, CandidateResult], None] | None = None,
    ) -> LoopResult:
        """Run the loop on ``splits`` (keys from :data:`SPLITS`).

        ``on_round`` is called after each adopted round, ``on_candidate`` after
        each candidate is scored (with the round number).
        """
        splits = self._check_splits(splits)
        history = History(metric=self.metric)

        removed: list[str] = []
        current = self._apply(splits, removed)
        model = self._fit(current)
        self._record(history, model, current, removed, on_round=on_round)

        stop_reason = STOP_MAX_ROUNDS
        for round_index in range(1, self.max_rounds + 1):
            ranking = self._rank(model, current[self.explain_split])
            excluded = set(removed)
            eligible = [(feature, score) for feature, score in ranking if feature not in excluded]
            if not eligible:
                logger.info("Round %d: no candidates left.", round_index)
                stop_reason = STOP_NO_CANDIDATES
                break

            parent = removed.copy()

            def evaluate(feature: str, parent: list[str] = parent) -> tuple[float, Any]:
                trial = self._apply(splits, [*parent, feature], SEARCH_SPLITS)
                fitted = self._fit(trial)
                return self._score(fitted, trial["verification"]), fitted

            def report(candidate: CandidateResult, round_index: int = round_index) -> None:
                logger.debug(
                    "Round %d candidate %d %r: %s=%.6f gain=%+.6f",
                    round_index,
                    candidate.rank,
                    candidate.feature,
                    self.metric.name,
                    candidate.score,
                    candidate.gain,
                )
                if on_candidate is not None:
                    on_candidate(round_index, candidate)

            result = search_candidates(
                eligible,
                evaluate,
                reference=history.rounds[-1].score("verification", self.metric.name),
                metric=self.metric,
                window=self.window,
                max_window=self.max_window,
                extension_step=self.extension_step,
                on_candidate=report,
            )
            removed.append(result.winner.feature)
            model = result.model
            current = self._apply(splits, removed)
            self._record(
                history,
                model,
                current,
                removed,
                selected=result.winner.feature,
                candidates=result.candidates,
                ranking=eligible[: self.ranking_head],
                search_stop=result.stop,
                on_round=on_round,
            )
            if self.stopping_rule.should_stop(history):
                stop_reason = STOP_RULE
                break

        history.stop_reason = stop_reason
        selected = self.stopping_rule.select(history)
        if not 0 <= selected < len(history):
            raise ValueError(f"Stopping rule selected round {selected}, outside the history.")
        history.selected_round = selected
        chosen = history.rounds[selected]
        if selected != len(history) - 1:
            model = self._fit(self._apply(splits, list(chosen.removed), SEARCH_SPLITS))
        logger.info(
            "Selected round %d (%d features removed, verification %s=%.6f).",
            selected,
            len(chosen.removed),
            self.metric.name,
            chosen.score("verification", self.metric.name),
        )
        return LoopResult(
            history=history, selected_round=selected, removed=chosen.removed, model=model
        )

    def _check_splits(self, splits: Mapping[str, D]) -> dict[str, D]:
        unknown = set(splits) - set(SPLITS)
        if unknown:
            raise ValueError(
                f"Unknown split names {sorted(unknown)}; expected names from {SPLITS}."
            )
        missing = [name for name in REQUIRED_SPLITS if splits.get(name) is None]
        if missing:
            raise ValueError(f"Missing required splits: {missing}.")
        if splits.get(self.explain_split) is None:
            raise ValueError(f"The explain split {self.explain_split!r} is missing.")
        return {name: data for name, data in splits.items() if data is not None}

    def _apply(
        self, splits: Mapping[str, D], removed: Sequence[str], names: Sequence[str] = SPLITS
    ) -> dict[str, D]:
        selected = {name: splits[name] for name in names if name in splits}
        if not removed:
            return selected
        return {name: self.remover.remove(data, list(removed)) for name, data in selected.items()}

    def _fit(self, data: Mapping[str, D]) -> Any:
        return self.model.fit(data["train"], data.get("validation"), seed=self.seed)

    def _rank(self, model: Any, data: D) -> list[tuple[str, float]]:
        seen: set[str] = set()
        ranking: list[tuple[str, float]] = []
        for feature, score in self.explainer.rank(model, data):
            name = str(feature)
            if name not in seen:
                seen.add(name)
                ranking.append((name, float(score)))
        return ranking

    def _score(self, model: Any, data: D) -> float:
        values = model.evaluate(data)
        if self.metric.name not in values:
            raise KeyError(
                f"Model.evaluate() did not return {self.metric.name!r}; got {sorted(values)}."
            )
        value = float(values[self.metric.name])
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite verification {self.metric.name!r}.")
        return value

    def _record(
        self,
        history: History,
        model: Any,
        data: Mapping[str, D],
        removed: Sequence[str],
        *,
        selected: str | None = None,
        candidates: tuple[CandidateResult, ...] = (),
        ranking: Sequence[tuple[str, float]] = (),
        search_stop: str | None = None,
        on_round: Callable[[RoundRecord], None] | None,
    ) -> None:
        metrics = {
            name: {key: float(value) for key, value in model.evaluate(split).items()}
            for name, split in data.items()
        }
        if self.metric.name not in metrics["verification"]:
            raise KeyError(
                f"Model.evaluate() did not return {self.metric.name!r}; "
                f"got {sorted(metrics['verification'])}."
            )
        record = RoundRecord(
            round=len(history),
            removed=tuple(removed),
            selected_feature=selected,
            metrics=metrics,
            candidates=candidates,
            ranking=tuple(ranking),
            search_stop=search_stop,
        )
        history.append(record)
        logger.info(
            "Round %d: removed %r, verification %s=%.6f.",
            record.round,
            selected,
            self.metric.name,
            record.score("verification", self.metric.name),
        )
        if on_round is not None:
            on_round(record)
