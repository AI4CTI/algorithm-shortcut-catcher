"""Column screening applied before each Random Forest fit.

Two modes:

``"autogluon"``
    Reproduces the preprocessing of AutoGluon 1.4 (``AutoMLPipelineFeatureGenerator``)
    for all-numeric data, which the published experiments ran through. Columns are
    screened on train and validation together: constant columns are dropped,
    two-valued columns become 0/1 indicators, and duplicate columns are dropped.
    Duplicates are detected on a random sample of rows (500, then 3000, drawn with
    replacement and ``random_state=0``), so two columns that agree on the sample
    but not on all rows are treated as duplicates. Use this mode to reproduce
    the legacy results.

``"exact"``
    Drops constant columns and exact duplicates of earlier columns, checked on
    all training rows. Values are used as they are.

In both modes the kept columns stay in their original order, and missing values
are set to 0 when building the model matrix.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Screening = Literal["autogluon", "exact", "none"]
SCREENING_MODES: tuple[Screening, ...] = ("autogluon", "exact", "none")

# AutoGluon 1.4 constants (autogluon.features.generators.astype / drop_duplicates).
_BOOL_HEAD_ROWS = 500
_BOOL_MIN_ROWS = 1000
_DUP_SAMPLE_INIT = 500
_DUP_SAMPLE_FINAL = 3000
_DUP_SEED = 0


@dataclass(frozen=True)
class ColumnScreen:
    """Columns kept for training and how to encode them."""

    features: tuple[str, ...]
    bool_true_values: dict[str, float] = field(default_factory=dict)

    def matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Return the model matrix for ``X`` (float32, missing values set to 0)."""
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise KeyError(f"Columns missing from the data: {missing[:10]}.")
        frame = X.loc[:, list(self.features)]
        if self.bool_true_values:
            frame = frame.copy()
            for column, true_value in self.bool_true_values.items():
                # Any other value, including unseen ones, maps to 0, as in AutoGluon.
                frame[column] = (frame[column] == true_value).astype(np.int8)
        return frame.fillna(0).to_numpy(dtype=np.float32)


def screen_columns(
    train: pd.DataFrame, validation: pd.DataFrame | None, mode: Screening
) -> ColumnScreen:
    """Decide which columns to train on."""
    if mode == "none":
        return ColumnScreen(features=tuple(train.columns))
    if mode == "exact":
        return ColumnScreen(features=tuple(_exact_screen(train)))
    if mode == "autogluon":
        frames = [train] if validation is None else [train, validation.loc[:, train.columns]]
        return _autogluon_screen(pd.concat(frames, ignore_index=True))
    raise ValueError(f"Unknown screening mode {mode!r}; expected one of {SCREENING_MODES}.")


def _exact_screen(X: pd.DataFrame) -> list[str]:
    keep: list[str] = []
    seen: dict[bytes, list[np.ndarray]] = {}
    for column in X.columns:
        values = X[column]
        if values.nunique(dropna=False) <= 1:
            continue
        array = values.to_numpy(dtype=np.float64)
        key = hashlib.sha1(array.tobytes()).digest()
        if any(np.array_equal(array, other, equal_nan=True) for other in seen.get(key, ())):
            continue
        seen.setdefault(key, []).append(array)
        keep.append(column)
    return keep


def _autogluon_screen(X: pd.DataFrame) -> ColumnScreen:
    # 1. Two-valued columns become indicators of their larger value (NaN never "true").
    if len(X) > _BOOL_MIN_ROWS:
        head = X.head(_BOOL_HEAD_ROWS).nunique(dropna=False)
        bool_candidates = [c for c in X.columns if head[c] <= 2]
    else:
        bool_candidates = list(X.columns)
    bool_true: dict[str, float] = {}
    for column in bool_candidates:
        uniques = np.sort(X[column].unique())
        if len(uniques) == 2:
            true_value = uniques[0] if np.isnan(uniques[1]) else uniques[1]
            bool_true[column] = float(true_value)
    converted = X.copy()
    for column, true_value in bool_true.items():
        converted[column] = (converted[column] == true_value).astype(np.int8)

    # 2. Constant columns carry no information.
    kept = [c for c in converted.columns if len(converted[c].unique()) > 1]
    converted = converted.loc[:, kept]

    # 3. Duplicate columns, detected on row samples.
    if len(converted) > _DUP_SAMPLE_INIT:
        candidates = _sampled_duplicates(converted, keep=False, sample_size=_DUP_SAMPLE_INIT)
        pool = converted.loc[:, candidates]
    else:
        pool = converted
    dropped = set(_sampled_duplicates(pool, keep="first", sample_size=_DUP_SAMPLE_FINAL))
    features = tuple(c for c in kept if c not in dropped)
    return ColumnScreen(
        features=features,
        bool_true_values={c: v for c, v in bool_true.items() if c in features},
    )


def _sampled_duplicates(
    X: pd.DataFrame, *, keep: Literal["first"] | Literal[False], sample_size: int
) -> list[str]:
    """Columns that duplicate another column on a row sample (AutoGluon's numeric rule)."""
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=_DUP_SEED, replace=True)
    groups: dict[float, list[str]] = defaultdict(list)
    for column in X.columns:
        groups[round(float(X[column].sum()), 2)].append(column)
    duplicates: list[str] = []
    for columns in groups.values():
        if len(columns) <= 1:
            continue
        kept = set(X[columns].T.drop_duplicates(keep=keep).T.columns)
        duplicates += [c for c in columns if c not in kept]
    return duplicates
