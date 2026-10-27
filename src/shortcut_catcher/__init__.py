"""ShortcutCatcher: find and remove shortcut features with an independent verification set."""

from .core import (
    FixedRounds,
    History,
    LoopResult,
    Metric,
    Patience,
    ShortcutCatcher,
)

__version__ = "0.1.0"

__all__ = [
    "FixedRounds",
    "History",
    "LoopResult",
    "Metric",
    "Patience",
    "ShortcutCatcher",
    "__version__",
]
