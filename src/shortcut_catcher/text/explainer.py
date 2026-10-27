"""Word ranking from ELECTRA pooling weights (experimental).

Ported as it stands from the legacy text implementation:

1. **Token weights** are the softmax pooling weights of the trained head.
2. **Token -> word**: words are ``\\b\\w+\\b`` matches (casefolded), aligned to
   tokens through the tokenizer offsets; a word's score is the ``max`` (or
   ``sum`` / ``avg``) of its token weights.
3. **Outliers per sample**, log-MAD rule: with ``r = w / mean(w)`` over the
   words of one sample, a word is an outlier if
   ``log r > median(log r) + z * 1.4826 * MAD(log r)`` and ``r >= min_ratio``.
   Samples with fewer than ``min_words`` words are not assessed. The legacy
   ``top_fraction`` rule is also available.
4. **Global ranking** over the explained split:

   * ``"count"`` (default): number of samples in which the word is an outlier
     (one vote per sample); ties by outlier rate, then assessed samples, then word.
   * ``"v8"``: keeps words with document frequency >= ``min_df``, normalised class
     entropy of their outlier score <= ``max_entropy`` and at least
     ``min_outliers`` outlier samples, then sorts them by summed outlier score
     (``v8_score_field``); ties keep the Count order.

Only words with a positive ranking score are returned as candidates.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import TextData
from .model import FittedElectra

WORD_PATTERN = re.compile(r"(?u)\b\w+\b")
TOKEN_TO_WORD = ("max", "sum", "avg")
OUTLIER_METHODS = ("log_mad", "top_fraction")
RANKINGS = ("count", "v8")
V8_SCORE_FIELDS = ("outlier_score", "outlier_sum_score")


def aligned_word_tokens(
    text: str, offsets: Sequence[tuple[int, int]]
) -> Iterator[tuple[re.Match[str], list[int]]]:
    """Yield each visible word with the indices of the tokens that overlap it."""
    valid = [(i, int(a), int(b)) for i, (a, b) in enumerate(offsets) if b > a]
    cursor = 0
    for match in WORD_PATTERN.finditer(text):
        while cursor < len(valid) and valid[cursor][2] <= match.start():
            cursor += 1
        indices = []
        k = cursor
        while k < len(valid) and valid[k][1] < match.end():
            if valid[k][2] > match.start():
                indices.append(valid[k][0])
            k += 1
        if indices:
            yield match, indices


def normalized_entropy(mass: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row of non-negative masses, divided by log(n_classes)."""
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 2 or mass.shape[1] < 2 or not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("Entropy needs finite non-negative masses over at least two classes.")
    peak = mass.max(axis=1, keepdims=True)
    scaled = np.divide(mass, peak, out=np.zeros_like(mass), where=peak > 0)
    total = scaled.sum(axis=1, keepdims=True)
    p = np.divide(scaled, total, out=np.zeros_like(mass), where=total > 0)
    logs = np.log(p, out=np.zeros_like(p), where=p > 0)
    entropy = np.clip(-(p * logs).sum(axis=1) / np.log(mass.shape[1]), 0, 1)
    entropy[entropy == 0] = 0.0
    return np.where(total[:, 0] > 0, entropy, np.nan)


@dataclass(frozen=True)
class WordReport:
    """Per-word statistics of one explanation."""

    words: pd.DataFrame
    """One row per word occurrence with its score and outlier flag."""
    general: pd.DataFrame
    """One row per word, sorted by the ranking; ``score`` holds the ranking value."""
    by_label: pd.DataFrame
    """The same statistics per (label, word)."""


class PoolingExplainer:
    """Rank words by how often their pooling weight is an outlier in a sample."""

    def __init__(
        self,
        *,
        ranking: str = "count",
        token_to_word: str = "max",
        outlier_method: str = "log_mad",
        outlier_z: float = 3.5,
        outlier_min_ratio: float = 2.0,
        outlier_min_words: int = 3,
        top_fraction: float = 0.25,
        min_df: float = 0.01,
        max_entropy: float = 0.5,
        min_outliers: int = 1,
        v8_score_field: str = "outlier_score",
    ) -> None:
        if ranking not in RANKINGS:
            raise ValueError(f"ranking must be one of {RANKINGS}.")
        if token_to_word not in TOKEN_TO_WORD:
            raise ValueError(f"token_to_word must be one of {TOKEN_TO_WORD}.")
        if outlier_method not in OUTLIER_METHODS:
            raise ValueError(f"outlier_method must be one of {OUTLIER_METHODS}.")
        if ranking == "v8" and outlier_method != "log_mad":
            raise ValueError("The v8 ranking needs outlier_method='log_mad'.")
        if not (math.isfinite(outlier_z) and outlier_z > 0):
            raise ValueError("outlier_z must be positive.")
        if not (math.isfinite(outlier_min_ratio) and outlier_min_ratio > 1):
            raise ValueError("outlier_min_ratio must be > 1.")
        if outlier_min_words < 3:
            raise ValueError("outlier_min_words must be at least 3.")
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1].")
        if not (0 <= min_df <= 1 and 0 <= max_entropy <= 1) or min_outliers < 1:
            raise ValueError("Invalid min_df, max_entropy or min_outliers.")
        if v8_score_field not in V8_SCORE_FIELDS:
            raise ValueError(f"v8_score_field must be one of {V8_SCORE_FIELDS}.")
        self.ranking = ranking
        self.token_to_word = token_to_word
        self.outlier_method = outlier_method
        self.outlier_z = float(outlier_z)
        self.outlier_min_ratio = float(outlier_min_ratio)
        self.outlier_min_words = int(outlier_min_words)
        self.top_fraction = float(top_fraction)
        self.min_df = float(min_df)
        self.max_entropy = float(max_entropy)
        self.min_outliers = int(min_outliers)
        self.v8_score_field = v8_score_field

    def rank(self, model: FittedElectra, data: TextData) -> list[tuple[str, float]]:
        report = self.explain(model, data)
        ranked = report.general[report.general["score"] > 0]
        return [(str(w), float(s)) for w, s in zip(ranked["word"], ranked["score"], strict=True)]

    def explain(self, model: FittedElectra, data: TextData) -> WordReport:
        """Compute word scores, outlier flags and the ranked word table."""
        rows: list[dict[str, Any]] = []
        for sample_id, (text, label, (offsets, weights)) in enumerate(
            zip(data.texts, data.labels, model.token_weights(data.texts), strict=True)
        ):
            for occurrence, (match, indices) in enumerate(aligned_word_tokens(text, offsets)):
                values = [float(weights[i]) for i in indices]
                if self.token_to_word == "max":
                    score = max(values)
                elif self.token_to_word == "sum":
                    score = sum(values)
                else:
                    score = sum(values) / len(values)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "word_occurrence_index": occurrence,
                        "word": match.group(0).casefold(),
                        "score": score,
                        "label": label,
                    }
                )
        words = self._mark_outliers(
            pd.DataFrame(
                rows, columns=["sample_id", "word_occurrence_index", "word", "score", "label"]
            )
        )
        samples = _word_samples(words)
        general = self._report(samples, [])
        by_label = self._report(samples, ["label"])
        general = self._count_order(general)
        if self.ranking == "v8":
            counts = pd.Series(data.labels).astype(str).value_counts()
            general = self._v8(general, by_label, counts)
        return WordReport(words=words, general=general.reset_index(drop=True), by_label=by_label)

    def _mark_outliers(self, words: pd.DataFrame) -> pd.DataFrame:
        words = words.copy()
        words["outlier_scored"] = False
        words["is_outlier"] = False
        words["attention_ratio"] = np.nan
        if words.empty:
            return words
        weights = words["score"].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Word scores must be finite and non-negative.")
        scored = np.zeros(len(words), dtype=bool)
        flags = np.zeros(len(words), dtype=bool)
        ratios = np.full(len(words), np.nan)
        for indices in words.groupby("sample_id", sort=False).indices.values():
            a = weights[indices]
            if self.outlier_method == "log_mad":
                if len(a) < self.outlier_min_words or a.max() <= 0:
                    continue  # not assessed
                scaled = a / a.max()
                ratio = scaled / scaled.mean()
                log_ratio = np.log(np.maximum(ratio, np.finfo(float).tiny))
                center = np.median(log_ratio)
                mad = np.median(np.abs(log_ratio - center))
                fence = center + self.outlier_z * 1.4826 * mad
                scored[indices] = True
                flags[indices] = (log_ratio > fence) & (ratio >= self.outlier_min_ratio)
                ratios[indices] = ratio
            else:
                scored[indices] = True
                count = max(1, math.ceil(len(a) * self.top_fraction))
                top = np.argsort(-a, kind="stable")[:count]
                chosen = top[a[top] > a.mean()]
                flags[indices[chosen]] = True
        words["outlier_scored"] = scored
        words["is_outlier"] = flags
        words["attention_ratio"] = ratios
        return words

    def _report(self, samples: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        group = [*keys, "word"]
        columns = [
            *group,
            "sample_count",
            "occurrence_count",
            "sum_score",
            "scored_sample_count",
            "outlier_sample_count",
            "outlier_sum_score",
            "outlier_score",
            "outlier_occurrence_count",
        ]
        if samples.empty:
            return pd.DataFrame(columns=[*columns, "avg_score", "outlier_rate"])
        sums = {
            k: (k, "sum")
            for k in (
                "occurrence_count",
                "sum_score",
                "scored_sample_count",
                "outlier_sample_count",
                "outlier_sum_score",
                "outlier_score",
                "outlier_occurrence_count",
            )
        }
        report = (
            samples.groupby(group, sort=False, dropna=False)
            .agg(sample_count=("sample_id", "size"), **sums)
            .reset_index()
        )
        for k in ("scored_sample_count", "outlier_sample_count", "outlier_occurrence_count"):
            report[k] = report[k].astype(int)
        report["avg_score"] = report.sum_score / report.occurrence_count
        report["outlier_rate"] = report.outlier_sample_count / report.scored_sample_count.replace(
            0, np.nan
        )
        if self.outlier_method != "log_mad":
            report["outlier_score"] = np.nan
        return report

    def _count_order(self, report: pd.DataFrame) -> pd.DataFrame:
        report = report.copy()
        report["score"] = report["outlier_sample_count"].astype(float)
        if self.outlier_method == "log_mad":
            order = ["score", "outlier_rate", "scored_sample_count", "word"]
        else:
            order = ["score", "sample_count", "word"]
        report = report.sort_values(
            order, ascending=[False] * (len(order) - 1) + [True], na_position="last"
        ).reset_index(drop=True)
        report["count_rank"] = np.arange(1, len(report) + 1)
        return report

    def _v8(self, general: pd.DataFrame, by_label: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
        if len(counts) < 2:
            raise ValueError("The v8 ranking needs at least two classes.")
        result = general.copy()
        field = self.v8_score_field
        result["document_frequency"] = result.sample_count / counts.sum()
        strength = result[field].astype(float)
        if result.empty:
            result["entropy"] = []
        else:
            masses = (
                by_label.assign(label=by_label.label.astype(str))
                .pivot(index="word", columns="label", values=field)
                .reindex(index=result.word, columns=counts.index)
                .fillna(0.0)
            )
            result["entropy"] = normalized_entropy(masses.to_numpy())
        eligible = (
            result.document_frequency.ge(self.min_df)
            & result.entropy.le(self.max_entropy)
            & result.outlier_sample_count.ge(self.min_outliers)
            & strength.gt(0)
        )
        result["eligible"] = eligible
        result["score"] = strength.where(eligible, 0.0)
        return result.sort_values(["score", "count_rank", "word"], ascending=[False, True, True])


def _word_samples(words: pd.DataFrame) -> pd.DataFrame:
    """Reduce repeated occurrences to one row per (sample, word)."""
    if words.empty:
        return words.assign(
            occurrence_count=[],
            sum_score=[],
            scored_sample_count=[],
            outlier_sample_count=[],
            outlier_sum_score=[],
            outlier_score=[],
            outlier_occurrence_count=[],
        )
    words = words.assign(
        flagged_weight=words.score.where(words.is_outlier),
        flagged_ratio=words.attention_ratio.where(words.is_outlier, 0.0),
    )
    result = (
        words.groupby(["sample_id", "word"], sort=False, dropna=False)
        .agg(
            occurrence_count=("score", "size"),
            sum_score=("score", "sum"),
            scored_sample_count=("outlier_scored", "max"),
            outlier_sample_count=("is_outlier", "max"),
            outlier_sum_score=("flagged_weight", "max"),
            outlier_score=("flagged_ratio", "max"),
            outlier_occurrence_count=("is_outlier", "sum"),
            label=("label", "first"),
        )
        .reset_index()
    )
    result["outlier_sum_score"] = result.outlier_sum_score.fillna(0.0)
    result["outlier_score"] = result.outlier_score.fillna(0.0)
    return result
