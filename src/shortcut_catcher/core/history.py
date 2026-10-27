"""Per-round records of a ShortcutCatcher run, with JSON and CSV export."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .metrics import Metric

#: Separator used for the cumulative removal list in CSV files.
CSV_LIST_SEPARATOR = ";"


@dataclass(frozen=True)
class CandidateResult:
    """Outcome of retraining without one candidate feature."""

    rank: int
    """1-based position among the eligible candidates of this round."""
    feature: str
    explainer_score: float
    score: float
    """Verification metric of the model retrained without this feature."""
    gain: float
    """Improvement over the parent round's verification metric; positive is better."""
    selected: bool = False


@dataclass(frozen=True)
class RoundRecord:
    """State adopted at the end of one round. Round 0 is the baseline."""

    round: int
    removed: tuple[str, ...]
    """All features removed so far, in removal order."""
    selected_feature: str | None
    metrics: dict[str, dict[str, float]]
    """Metric values per split: ``{"verification": {"f1_macro": 0.7}, ...}``."""
    candidates: tuple[CandidateResult, ...] = ()
    ranking: tuple[tuple[str, float], ...] = ()
    """Head of the explainer ranking the candidates were drawn from."""
    search_stop: str | None = None

    def score(self, split: str, metric: str) -> float:
        """Return one metric value, or NaN when it was not recorded."""
        return self.metrics.get(split, {}).get(metric, math.nan)


@dataclass
class History:
    """Ordered round records plus the run outcome."""

    metric: Metric
    rounds: list[RoundRecord] = field(default_factory=list)
    stop_reason: str | None = None
    selected_round: int | None = None

    def append(self, record: RoundRecord) -> None:
        if record.round != len(self.rounds):
            raise ValueError(f"Expected round {len(self.rounds)}, got {record.round}.")
        self.rounds.append(record)

    def __len__(self) -> int:
        return len(self.rounds)

    def __iter__(self) -> Iterator[RoundRecord]:
        return iter(self.rounds)

    def __getitem__(self, index: int) -> RoundRecord:
        return self.rounds[index]

    @property
    def selected(self) -> RoundRecord | None:
        """The selected round record, once the run has finished."""
        return None if self.selected_round is None else self.rounds[self.selected_round]

    def scores(self, split: str = "verification", metric: str | None = None) -> list[float]:
        """Return one metric across rounds."""
        name = metric or self.metric.name
        return [record.score(split, name) for record in self.rounds]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": asdict(self.metric),
            "stop_reason": self.stop_reason,
            "selected_round": self.selected_round,
            "rounds": [_round_to_dict(record) for record in self.rounds],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> History:
        history = cls(
            metric=Metric(**payload["metric"]),
            stop_reason=payload.get("stop_reason"),
            selected_round=payload.get("selected_round"),
        )
        for item in payload["rounds"]:
            history.append(_round_from_dict(item))
        return history

    def write(self, directory: str | Path) -> dict[str, Path]:
        """Write ``history.json``, ``rounds.csv`` and ``candidates.csv`` to ``directory``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "history": out / "history.json",
            "rounds": out / "rounds.csv",
            "candidates": out / "candidates.csv",
        }
        paths["history"].write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        self._write_rounds_csv(paths["rounds"])
        self._write_candidates_csv(paths["candidates"])
        return paths

    def _write_rounds_csv(self, path: Path) -> None:
        columns = sorted({(s, m) for r in self.rounds for s, ms in r.metrics.items() for m in ms})
        header = ["round", "selected_feature", "n_removed", "removed", "selected"]
        header += [f"{split}_{metric}" for split, metric in columns]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for record in self.rounds:
                writer.writerow(
                    [
                        record.round,
                        record.selected_feature or "",
                        len(record.removed),
                        CSV_LIST_SEPARATOR.join(record.removed),
                        record.round == self.selected_round,
                        *(record.score(split, metric) for split, metric in columns),
                    ]
                )

    def _write_candidates_csv(self, path: Path) -> None:
        header = ["round", "rank", "feature", "explainer_score", "score", "gain", "selected"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for record in self.rounds:
                for c in record.candidates:
                    writer.writerow(
                        [
                            record.round,
                            c.rank,
                            c.feature,
                            c.explainer_score,
                            c.score,
                            c.gain,
                            c.selected,
                        ]
                    )


def _round_to_dict(record: RoundRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["removed"] = list(record.removed)
    payload["ranking"] = [list(item) for item in record.ranking]
    return payload


def _round_from_dict(item: dict[str, Any]) -> RoundRecord:
    return RoundRecord(
        round=int(item["round"]),
        removed=tuple(item["removed"]),
        selected_feature=item.get("selected_feature"),
        metrics={split: dict(values) for split, values in item["metrics"].items()},
        candidates=tuple(CandidateResult(**c) for c in item.get("candidates", ())),
        ranking=tuple((str(f), float(s)) for f, s in item.get("ranking", ())),
        search_stop=item.get("search_stop"),
    )
