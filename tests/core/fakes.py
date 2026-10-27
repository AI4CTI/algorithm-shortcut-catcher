"""Toy implementations of the four core protocols.

A dataset is a split name plus the set of features still present. A model's
score on a split is a user-supplied function of the features it was trained on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

ScoreFn = Callable[[frozenset[str]], float]


@dataclass(frozen=True)
class ToyData:
    split: str
    features: frozenset[str]


@dataclass
class ToyFitted:
    features: frozenset[str]
    seed: int
    scores: Mapping[str, ScoreFn]

    def evaluate(self, data: ToyData) -> dict[str, float]:
        value = self.scores[data.split](self.features & data.features)
        return {"f1_macro": value, "nll": 1.0 - value}


@dataclass
class ToyModel:
    scores: Mapping[str, ScoreFn]
    fits: list[tuple[frozenset[str], int, bool]] = field(default_factory=list)

    def fit(self, train: ToyData, validation: ToyData | None, *, seed: int) -> ToyFitted:
        self.fits.append((train.features, seed, validation is not None))
        return ToyFitted(train.features, seed, self.scores)


@dataclass
class ToyExplainer:
    importance: Mapping[str, float]
    positive_only: bool = False

    def rank(self, model: ToyFitted, data: ToyData) -> list[tuple[str, float]]:
        ranked = sorted(model.features, key=lambda f: (-self.importance[f], f))
        return [
            (f, self.importance[f])
            for f in ranked
            if not self.positive_only or self.importance[f] > 0
        ]


class ToyRemover:
    def remove(self, data: ToyData, features: Sequence[str]) -> ToyData:
        return ToyData(data.split, data.features - set(features))


FEATURES = frozenset({"leak", "b", "c", "d", "e", "f"})
IMPORTANCE = {"leak": 10.0, "b": 5.0, "c": 4.0, "d": 3.0, "e": 2.0, "f": 1.0}
GOOD = ("b", "c", "d", "e", "f")


def shortcut_scores() -> dict[str, ScoreFn]:
    """``leak`` helps on IID splits and hurts on verification/check; others help a little."""

    def iid(features: frozenset[str]) -> float:
        return 0.5 + 0.05 * len(features & set(GOOD)) + 0.2 * ("leak" in features)

    def ood(features: frozenset[str]) -> float:
        return 0.5 + 0.05 * len(features & set(GOOD)) - 0.3 * ("leak" in features)

    return {"train": iid, "validation": iid, "test": iid, "verification": ood, "check": ood}


def make_splits(names: Sequence[str] = ("train", "validation", "test", "verification", "check")):
    return {name: ToyData(name, FEATURES) for name in names}
