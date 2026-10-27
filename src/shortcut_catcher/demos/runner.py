"""Bundled demos: synthetic data with several planted shortcuts, run end to end."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core import CandidateResult, LoopResult, RoundRecord, ShortcutCatcher

DEMO_NAMES = ("tabular", "text")
#: Seed used when none is given; these produce the runs described in the README.
DEFAULT_SEEDS = {"tabular": 0, "text": 2}


@dataclass(frozen=True)
class DemoOutcome:
    """A demo run and which planted shortcuts it removed."""

    name: str
    seed: int
    shortcuts: tuple[str, ...]
    metric: str
    result: LoopResult

    @property
    def removal_rounds(self) -> dict[str, int | None]:
        """Round in which each planted shortcut was removed (None if never)."""
        rounds: dict[str, int | None] = dict.fromkeys(self.shortcuts)
        for record in self.result.history:
            if record.selected_feature in rounds:
                rounds[record.selected_feature] = record.round
        return rounds

    @property
    def found(self) -> tuple[str, ...]:
        """Planted shortcuts removed in the selected round."""
        return tuple(s for s in self.shortcuts if s in self.result.removed)

    @property
    def missed(self) -> tuple[str, ...]:
        return tuple(s for s in self.shortcuts if s not in self.result.removed)

    @property
    def other_removed(self) -> tuple[str, ...]:
        """Removed features in the selected round that were not planted shortcuts."""
        return tuple(f for f in self.result.removed if f not in self.shortcuts)

    @property
    def baseline_score(self) -> float:
        return self.result.history[0].score("verification", self.metric)

    @property
    def selected_score(self) -> float:
        return self.result.history[self.result.selected_round].score("verification", self.metric)

    def ranks_before_round_1(self) -> dict[str, int | None]:
        """Position of each planted shortcut in the first explainer ranking."""
        if len(self.result.history) < 2:
            return dict.fromkeys(self.shortcuts)
        order = [f for f, _ in self.result.history[1].ranking]
        return {s: (order.index(s) + 1 if s in order else None) for s in self.shortcuts}


def run_demo(
    name: str,
    *,
    seed: int | None = None,
    on_round: Callable[[RoundRecord], None] | None = None,
    on_candidate: Callable[[int, CandidateResult], None] | None = None,
) -> DemoOutcome:
    """Run the ``tabular`` or ``text`` demo; ``seed`` defaults to :data:`DEFAULT_SEEDS`."""
    if name not in DEMO_NAMES:
        raise ValueError(f"Unknown demo {name!r}; choose from {DEMO_NAMES}.")
    seed = DEFAULT_SEEDS[name] if seed is None else seed
    metric = "f1_macro"
    loop: ShortcutCatcher[Any]
    if name == "tabular":
        from ..tabular import ColumnRemover, GiniExplainer, RandomForestModel
        from .tabular import SHORTCUTS as tabular_shortcuts
        from .tabular import make_shortcut_splits as tabular_splits

        shortcuts = tabular_shortcuts
        splits: dict[str, Any] = dict(tabular_splits(seed=seed))
        loop = ShortcutCatcher(
            RandomForestModel(),
            GiniExplainer(),
            ColumnRemover(),
            metric=metric,
            max_rounds=6,
            window=5,
            seed=seed,
        )
    else:
        from ..text import ElectraModel, PoolingExplainer, WordRemover
        from .text import SHORTCUTS as text_shortcuts
        from .text import make_shortcut_splits as text_splits

        shortcuts = text_shortcuts
        splits = dict(text_splits(seed=seed))
        loop = ShortcutCatcher(
            ElectraModel(epochs=20, batch_size=32, device="cpu"),
            PoolingExplainer(),
            WordRemover(),
            metric=metric,
            max_rounds=3,
            window=5,
            seed=seed,
        )
    result = loop.run(splits, on_round=on_round, on_candidate=on_candidate)
    return DemoOutcome(name=name, seed=seed, shortcuts=shortcuts, metric=metric, result=result)
