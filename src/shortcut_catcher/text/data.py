"""Text datasets and whole-word removal."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TextData:
    """Texts and their labels."""

    texts: tuple[str, ...]
    labels: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError(f"{len(self.texts)} texts but {len(self.labels)} labels.")

    def __len__(self) -> int:
        return len(self.texts)

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, text_column: str = "text", label_column: str = "label"
    ) -> TextData:
        missing = {text_column, label_column} - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}.")
        if frame[[text_column, label_column]].isna().any().any():
            raise ValueError("Texts and labels must not be missing.")
        return cls(
            texts=tuple(str(t) for t in frame[text_column]),
            labels=tuple(frame[label_column].tolist()),
        )

    def with_texts(self, texts: Sequence[str]) -> TextData:
        return TextData(texts=tuple(texts), labels=self.labels)


def load_text_splits(
    source: str | Path | Mapping[str, str | Path],
    *,
    text_column: str = "text",
    label_column: str = "label",
    split_column: str = "split",
) -> dict[str, TextData]:
    """Read splits from one file with a split column, or from one file per split (CSV or Parquet)."""
    if isinstance(source, Mapping):
        return {
            name: TextData.from_frame(_read(path), text_column, label_column)
            for name, path in source.items()
        }
    frame = _read(source)
    if split_column not in frame.columns:
        raise ValueError(f"Column {split_column!r} not found in {source}.")
    return {
        str(name): TextData.from_frame(group, text_column, label_column)
        for name, group in frame.groupby(split_column, sort=False)
    }


def _read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=False)


def _word_pattern(words: Iterable[str]) -> re.Pattern[str] | None:
    clean = sorted(
        {str(w).strip() for w in words if str(w).strip()}, key=lambda w: (-len(w), w.casefold())
    )
    if not clean:
        return None
    alternatives = "|".join(re.escape(w) for w in clean)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


def _delete(text: str, pattern: re.Pattern[str]) -> str:
    text = pattern.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def delete_words(text: str, words: Iterable[str]) -> str:
    """Delete case-insensitive whole-word matches and tidy the surrounding whitespace."""
    pattern = _word_pattern(words)
    return text if pattern is None else _delete(text, pattern)


class WordRemover:
    """Remove words from every text (case-insensitive, whole words only)."""

    def remove(self, data: TextData, features: Sequence[str]) -> TextData:
        pattern = _word_pattern(features)
        if pattern is None:
            return data
        return data.with_texts([_delete(t, pattern) for t in data.texts])
