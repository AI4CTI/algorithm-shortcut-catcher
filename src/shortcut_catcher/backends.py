"""Built-in backends and the code that runs a configuration end to end."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ConfigError, DataConfig, LoopConfig, RunConfig
from .core import (
    SPLITS,
    CandidateResult,
    FixedRounds,
    LoopResult,
    Patience,
    RoundRecord,
    ShortcutCatcher,
)


@dataclass(frozen=True)
class BackendInfo:
    name: str
    description: str
    experimental: bool
    requires: tuple[str, ...]
    install: str

    @property
    def available(self) -> bool:
        return all(importlib.util.find_spec(module) is not None for module in self.requires)


BACKENDS: dict[str, BackendInfo] = {
    "tabular": BackendInfo(
        name="tabular",
        description="Random Forest (scikit-learn), Gini importance, column removal.",
        experimental=False,
        requires=("sklearn",),
        install="uv sync (or, from the repository: pip install .)",
    ),
    "text": BackendInfo(
        name="text",
        description="Frozen ELECTRA encoder, pooling-weight word ranking, word removal.",
        experimental=True,
        requires=("torch", "transformers"),
        install='uv sync --extra text (or, from the repository: pip install ".[text]")',
    ),
}


def backend_info(name: str) -> BackendInfo:
    if name not in BACKENDS:
        raise ConfigError(f"Unknown backend {name!r}; available: {sorted(BACKENDS)}.")
    info = BACKENDS[name]
    if not info.available:
        raise ConfigError(f"Backend {name!r} is not installed. Install it with: {info.install}")
    return info


def build_components(
    backend: str, model: Mapping[str, Any], explainer: Mapping[str, Any]
) -> tuple[Any, Any, Any]:
    """Return (model, explainer, remover) for a backend."""
    backend_info(backend)
    try:
        if backend == "tabular":
            from .tabular import ColumnRemover, GiniExplainer, RandomForestModel

            if explainer:
                raise ConfigError("The tabular Gini explainer takes no options.")
            return RandomForestModel(**model), GiniExplainer(), ColumnRemover()
        from .text import ElectraModel, PoolingExplainer, WordRemover

        return ElectraModel(**model), PoolingExplainer(**explainer), WordRemover()
    except (TypeError, ValueError) as error:
        if isinstance(error, ConfigError):
            raise
        raise ConfigError(f"Invalid {backend} option: {error}") from error


def load_splits(backend: str, data: DataConfig) -> dict[str, Any]:
    """Read the data files of a configuration into backend datasets."""
    # Text keeps every string ("NA", "None" are texts); tabular reads them as missing.
    keep_default_na = backend != "text"
    frames: dict[str, pd.DataFrame]
    if data.file is not None:
        frame = _read_table(data.file, keep_default_na=keep_default_na)
        if data.split_column not in frame.columns:
            raise ConfigError(f"Column {data.split_column!r} not found in {data.file}.")
        frames = {
            str(name): group.drop(columns=[data.split_column]).reset_index(drop=True)
            for name, group in frame.groupby(data.split_column, sort=False)
        }
        unknown = set(frames) - set(SPLITS)
        if unknown:
            raise ConfigError(f"Unknown values in {data.split_column!r}: {sorted(unknown)}.")
    else:
        frames = {
            name: _read_table(path, keep_default_na=keep_default_na)
            for name, path in data.splits.items()
        }

    if backend == "tabular":
        from .tabular import TabularData
        from .tabular.data import align_columns

        try:
            return align_columns(
                {n: TabularData.from_frame(f, data.label_column) for n, f in frames.items()}
            )
        except ValueError as error:
            raise ConfigError(str(error)) from error
    from .text import TextData

    return {
        n: TextData.from_frame(f, data.text_column, data.label_column) for n, f in frames.items()
    }


def _read_table(path: Path, *, keep_default_na: bool = True) -> pd.DataFrame:
    if not path.is_file():
        raise ConfigError(f"Data file not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=keep_default_na)


def make_loop(loop: LoopConfig, components: tuple[Any, Any, Any]) -> ShortcutCatcher[Any]:
    """Build the loop from its settings."""
    if loop.stopping == "patience":
        assert loop.patience is not None
        rule: FixedRounds = Patience(loop.patience, metric=loop.selection_metric)
    else:
        rule = FixedRounds(metric=loop.selection_metric)
    model, explainer, remover = components
    return ShortcutCatcher(
        model,
        explainer,
        remover,
        metric=loop.metric,
        max_rounds=loop.max_rounds,
        window=loop.window,
        max_window=loop.max_window,
        extension_step=loop.extension_step,
        stopping_rule=rule,
        seed=loop.seed,
    )


def run_config(
    config: RunConfig,
    *,
    on_round: Callable[[RoundRecord], None] | None = None,
    on_candidate: Callable[[int, CandidateResult], None] | None = None,
) -> LoopResult:
    """Load the data, build the components and run the loop."""
    components = build_components(config.backend, config.model, config.explainer)
    splits = load_splits(config.backend, config.data)
    loop = make_loop(config.loop, components)
    return loop.run(splits, on_round=on_round, on_candidate=on_candidate)


def summarize(result: LoopResult, metric: str) -> dict[str, Any]:
    """A JSON-ready summary of a run."""
    selected = result.history.rounds[result.selected_round]
    summary: dict[str, Any] = {
        "metric": metric,
        "selected_round": result.selected_round,
        "removed": list(result.removed),
        "stop_reason": result.history.stop_reason,
        "selected_metrics": selected.metrics,
        "rounds": [
            {
                "round": r.round,
                "removed": r.selected_feature,
                "verification": r.score("verification", metric),
                "check": r.score("check", metric),
            }
            for r in result.history
        ],
    }
    features = getattr(result.model, "features", None)
    if features is not None:
        summary["kept_features"] = list(features)
    return summary


def write_outputs(
    result: LoopResult, output_dir: Path, *, metric: str, config: Mapping[str, Any]
) -> dict[str, Path]:
    """Write the history, the selected feature set and the run configuration."""
    paths = result.history.write(output_dir)
    paths["selected"] = output_dir / "selected.json"
    paths["selected"].write_text(
        json.dumps(summarize(result, metric), indent=2, default=str) + "\n", encoding="utf-8"
    )
    paths["config"] = output_dir / "config.json"
    paths["config"].write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    return paths


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), default=str))  # type: ignore[no-any-return]
