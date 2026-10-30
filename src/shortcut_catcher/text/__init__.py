"""Text backend (experimental): frozen ELECTRA encoder, pooling-weight word ranking, word removal.

Requires the optional ``[text]`` extra: ``uv sync --extra text`` (or ``pip install ".[text]"``).
This backend is a port of research code and is marked experimental: its word
scoring has not been validated beyond the original experiments.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError as error:  # pragma: no cover - depends on the environment
    raise ImportError(
        "The text backend needs the optional [text] extra: "
        'uv sync --extra text (or, from the repository: pip install ".[text]").'
    ) from error

from .data import TextData, WordRemover, delete_words, load_text_splits
from .explainer import PoolingExplainer, WordReport
from .model import DEFAULT_CHECKPOINT, ElectraModel, FittedElectra

__all__ = [
    "DEFAULT_CHECKPOINT",
    "ElectraModel",
    "FittedElectra",
    "PoolingExplainer",
    "TextData",
    "WordRemover",
    "WordReport",
    "delete_words",
    "load_text_splits",
]
