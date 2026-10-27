"""Protocols a backend implements to run the ShortcutCatcher loop.

A backend chooses its own data type ``D`` (a DataFrame, a dict of arrays, a list
of texts, ...). The core never inspects ``D``; it only passes it between these
four components.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from .history import History

D = TypeVar("D")
D_contra = TypeVar("D_contra", contravariant=True)

#: A feature name and its importance score.
FeatureScore = tuple[str, float]


class FittedModel(Protocol[D_contra]):
    """A trained model that can be scored on a dataset."""

    def evaluate(self, data: D_contra) -> Mapping[str, float]:
        """Return metric values on ``data``, e.g. ``{"accuracy": 0.9, "f1_macro": 0.8}``."""
        ...


class Model(Protocol[D_contra]):
    """Trains a fresh model. Called once per candidate, so it must not keep state between fits."""

    def fit(
        self, train: D_contra, validation: D_contra | None, *, seed: int
    ) -> FittedModel[D_contra]:
        """Train on ``train``; ``validation`` may be used for checkpoint selection."""
        ...


class Explainer(Protocol[D_contra]):
    """Ranks the features a fitted model relies on."""

    def rank(self, model: Any, data: D_contra) -> Sequence[FeatureScore]:
        """Return eligible features as ``(name, score)`` pairs, most important first.

        ``model`` is the object returned by the backend's :meth:`Model.fit`.
        Features left out of the ranking are never removal candidates.
        """
        ...


class Remover(Protocol[D]):
    """Removes features from a dataset."""

    def remove(self, data: D, features: Sequence[str]) -> D:
        """Return a copy of ``data`` without ``features``. Must not modify ``data``."""
        ...


class StoppingRule(Protocol):
    """Decides when the loop stops early and which round is selected."""

    def should_stop(self, history: History) -> bool:
        """Return True to stop after the last recorded round."""
        ...

    def select(self, history: History) -> int:
        """Return the index of the selected round in ``history.rounds``."""
        ...
