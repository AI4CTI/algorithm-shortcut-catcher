"""Gini-importance explainer and column remover."""

from __future__ import annotations

from collections.abc import Sequence

from .data import TabularData
from .model import FittedForest


class GiniExplainer:
    """Rank features by the forest's impurity-based (Gini) importance.

    Only features the forest was trained on are ranked, so constant columns
    are never candidates. Ties keep the column order.
    """

    def rank(self, model: FittedForest, data: TabularData) -> list[tuple[str, float]]:
        scores = [float(v) for v in model.estimator.feature_importances_]
        pairs = list(zip(model.features, scores, strict=True))
        return sorted(pairs, key=lambda item: item[1], reverse=True)


class ColumnRemover:
    """Drop feature columns from every split."""

    def remove(self, data: TabularData, features: Sequence[str]) -> TabularData:
        return data.drop(features)
