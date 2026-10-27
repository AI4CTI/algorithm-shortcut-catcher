"""Tabular datasets: a feature table plus a label column."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TabularData:
    """Feature table ``X`` and labels ``y`` with the same row index."""

    X: pd.DataFrame
    y: pd.Series

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y):
            raise ValueError(f"X has {len(self.X)} rows but y has {len(self.y)}.")
        if self.X.columns.duplicated().any():
            duplicated = self.X.columns[self.X.columns.duplicated()].tolist()
            raise ValueError(f"Duplicated column names: {duplicated}.")
        non_numeric = [c for c in self.X.columns if not pd.api.types.is_numeric_dtype(self.X[c])]
        if non_numeric:
            raise ValueError(
                f"Non-numeric feature columns {non_numeric[:10]}; encode them before running."
            )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, label_column: str) -> TabularData:
        """Split ``frame`` into features and the ``label_column``."""
        if label_column not in frame.columns:
            raise ValueError(f"Label column {label_column!r} not found.")
        X = frame.drop(columns=[label_column])
        X.columns = [str(c) for c in X.columns]
        return cls(X=X, y=frame[label_column])

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)

    def drop(self, features: Iterable[str]) -> TabularData:
        """Return a copy without the given columns; unknown names are ignored."""
        present = [f for f in features if f in self.X.columns]
        if not present:
            return self
        return TabularData(X=self.X.drop(columns=present), y=self.y)


def align_columns(splits: Mapping[str, TabularData]) -> dict[str, TabularData]:
    """Check all splits have the same feature columns and order them like the first split."""
    reference_name, reference = next(iter(splits.items()))
    aligned = {}
    for name, data in splits.items():
        if set(data.features) != set(reference.features):
            missing = sorted(set(reference.features) - set(data.features))
            extra = sorted(set(data.features) - set(reference.features))
            raise ValueError(
                f"Split {name!r} columns differ from {reference_name!r}: "
                f"missing {missing[:10]}, extra {extra[:10]}."
            )
        aligned[name] = (
            data
            if data.features == reference.features
            else TabularData(X=data.X.loc[:, reference.features], y=data.y)
        )
    return aligned


def load_csv_splits(paths: Mapping[str, str | Path], label_column: str) -> dict[str, TabularData]:
    """Read one CSV per split; all splits must share the same feature columns."""
    return align_columns(
        {
            name: TabularData.from_frame(pd.read_csv(path), label_column)
            for name, path in paths.items()
        }
    )
