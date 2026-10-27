"""Random Forest model for the tabular backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from .data import TabularData
from .screening import SCREENING_MODES, ColumnScreen, Screening, screen_columns

#: Random Forest defaults of AutoGluon 1.4, used by the published experiments.
AUTOGLUON_RF_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_leaf_nodes": 15000,
    "criterion": "gini",
    "bootstrap": True,
    "n_jobs": -1,
}


@dataclass(frozen=True)
class FittedForest:
    """A trained forest and the column screen it was trained with."""

    estimator: RandomForestClassifier
    screen: ColumnScreen

    @property
    def features(self) -> tuple[str, ...]:
        """Columns the forest was trained on, in model order."""
        return self.screen.features

    def predict(self, data: TabularData) -> np.ndarray:
        return self.estimator.predict(self.screen.matrix(data.X))

    def evaluate(self, data: TabularData) -> dict[str, float]:
        """Return accuracy, macro/micro F1, macro precision/recall and NLL."""
        proba = self.estimator.predict_proba(self.screen.matrix(data.X))
        classes = self.estimator.classes_
        y = data.y.to_numpy()
        pred = classes[np.argmax(proba, axis=1)]
        kwargs: dict[str, Any] = {"zero_division": 0}
        known = np.isin(y, classes)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro", **kwargs)),
            "f1_micro": float(f1_score(y, pred, average="micro", **kwargs)),
            "precision_macro": float(precision_score(y, pred, average="macro", **kwargs)),
            "recall_macro": float(recall_score(y, pred, average="macro", **kwargs)),
            # NLL is defined only on samples whose class was seen in training.
            "nll": float(log_loss(y[known], proba[known], labels=classes))
            if known.any()
            else float("nan"),
        }


class RandomForestModel:
    """scikit-learn Random Forest with the settings of the published experiments.

    Args:
        screening: How columns are screened before each fit; see
            :mod:`shortcut_catcher.tabular.screening`. ``"autogluon"`` (default)
            reproduces the legacy AutoGluon pipeline; ``"exact"`` only drops
            constant columns and exact duplicates.
        **params: Passed to ``RandomForestClassifier``, overriding
            :data:`AUTOGLUON_RF_DEFAULTS`. The seed is given to :meth:`fit`.
    """

    def __init__(self, *, screening: Screening = "autogluon", **params: Any) -> None:
        if screening not in SCREENING_MODES:
            raise ValueError(f"screening must be one of {SCREENING_MODES}.")
        if "random_state" in params:
            raise ValueError("Pass the seed to fit(), not random_state.")
        unknown = set(params) - set(RandomForestClassifier().get_params())
        if unknown:
            raise ValueError(f"Unknown RandomForestClassifier options: {sorted(unknown)}.")
        self.screening: Screening = screening
        self.params = {**AUTOGLUON_RF_DEFAULTS, **params}

    def fit(self, train: TabularData, validation: TabularData | None, *, seed: int) -> FittedForest:
        """Train a fresh forest. ``validation`` only takes part in column screening."""
        screen = screen_columns(
            train.X, None if validation is None else validation.X, self.screening
        )
        if not screen.features:
            raise ValueError("No informative feature columns left to train on.")
        estimator = RandomForestClassifier(random_state=seed, **self.params)
        estimator.fit(screen.matrix(train.X), train.y.to_numpy())
        return FittedForest(estimator=estimator, screen=screen)
