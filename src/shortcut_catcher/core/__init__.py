"""Application-agnostic ShortcutCatcher algorithm."""

from .history import CandidateResult, History, RoundRecord
from .interfaces import Explainer, FeatureScore, FittedModel, Model, Remover, StoppingRule
from .loop import REQUIRED_SPLITS, SPLITS, LoopResult, ShortcutCatcher
from .metrics import Metric, as_metric
from .search import SearchResult, search_candidates
from .stopping import FixedRounds, Patience, best_round

__all__ = [
    "REQUIRED_SPLITS",
    "SPLITS",
    "CandidateResult",
    "Explainer",
    "FeatureScore",
    "FittedModel",
    "FixedRounds",
    "History",
    "LoopResult",
    "Metric",
    "Model",
    "Patience",
    "Remover",
    "RoundRecord",
    "SearchResult",
    "ShortcutCatcher",
    "StoppingRule",
    "as_metric",
    "best_round",
    "search_candidates",
]
