"""Run configuration: a YAML or TOML file turned into typed settings.

Example (YAML)::

    backend: tabular            # or: text (experimental, needs the [text] extra)
    data:
      label_column: class
      splits:                   # one file per split; or `file:` + `split_column:`
        train: data/train.csv
        validation: data/val.csv
        verification: data/verification.csv
        check: data/check.csv     # optional held-out split, never used for decisions
    loop:
      metric: f1_macro
      max_rounds: 10
      window: 5
      seed: 0
    model: {}                   # backend model options
    explainer: {}               # backend explainer options
    output_dir: runs/my-run

Relative paths are resolved against the directory of the configuration file.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .core import SPLITS


class ConfigError(ValueError):
    """Invalid run configuration."""


@dataclass(frozen=True)
class DataConfig:
    """Where the splits are and which columns hold the label (and text)."""

    splits: dict[str, Path] = field(default_factory=dict)
    file: Path | None = None
    split_column: str = "split"
    label_column: str = "label"
    text_column: str = "text"


@dataclass(frozen=True)
class LoopConfig:
    """Settings of the ShortcutCatcher loop."""

    metric: str = "f1_macro"
    max_rounds: int = 10
    window: int = 5
    max_window: int | None = None
    extension_step: int = 1
    seed: int = 0
    stopping: str = "fixed"
    patience: int | None = None
    selection_metric: str | None = None


@dataclass(frozen=True)
class RunConfig:
    """A complete run: backend, data, loop settings and backend options."""

    backend: str
    data: DataConfig
    loop: LoopConfig = field(default_factory=LoopConfig)
    model: dict[str, Any] = field(default_factory=dict)
    explainer: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None


def load_config(path: str | Path) -> RunConfig:
    """Read a ``.yaml``/``.yml`` or ``.toml`` file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        payload = yaml.safe_load(text)
    elif path.suffix == ".toml":
        if sys.version_info >= (3, 11):
            import tomllib
        else:  # pragma: no cover
            import tomli as tomllib
        payload = tomllib.loads(text)
    else:
        raise ConfigError("Use a .yaml, .yml or .toml configuration file.")
    if not isinstance(payload, Mapping):
        raise ConfigError("The configuration must be a mapping at the top level.")
    return parse_config(payload, base_dir=path.parent)


def parse_config(payload: Mapping[str, Any], base_dir: str | Path = ".") -> RunConfig:
    """Validate a configuration mapping; relative paths are resolved against ``base_dir``."""
    base = Path(base_dir)
    _check_keys(payload, {f.name for f in fields(RunConfig)}, "top level")
    if "backend" not in payload:
        raise ConfigError("Missing `backend` (tabular or text).")
    if "data" not in payload:
        raise ConfigError("Missing `data` section.")

    data_payload = dict(_mapping(payload["data"], "data"))
    _check_keys(data_payload, {f.name for f in fields(DataConfig)}, "data")
    splits = {
        str(name): _path(value, base)
        for name, value in _mapping(data_payload.pop("splits", {}) or {}, "data.splits").items()
    }
    unknown = set(splits) - set(SPLITS)
    if unknown:
        raise ConfigError(f"Unknown split names {sorted(unknown)}; use {list(SPLITS)}.")
    file = data_payload.pop("file", None)
    if bool(splits) == (file is not None):
        raise ConfigError("Give either `data.splits` (one file per split) or `data.file`.")
    data = DataConfig(
        splits=splits, file=None if file is None else _path(file, base), **data_payload
    )

    loop_payload = dict(_mapping(payload.get("loop", {}) or {}, "loop"))
    _check_keys(loop_payload, {f.name for f in fields(LoopConfig)}, "loop")
    loop = LoopConfig(**loop_payload)
    if loop.stopping not in ("fixed", "patience"):
        raise ConfigError("loop.stopping must be `fixed` or `patience`.")
    if loop.stopping == "patience" and loop.patience is None:
        raise ConfigError("loop.patience is required when loop.stopping is `patience`.")

    output_dir = payload.get("output_dir")
    return RunConfig(
        backend=str(payload["backend"]),
        data=data,
        loop=loop,
        model=dict(_mapping(payload.get("model", {}) or {}, "model")),
        explainer=dict(_mapping(payload.get("explainer", {}) or {}, "explainer")),
        output_dir=None if output_dir is None else _path(output_dir, base),
    )


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"`{where}` must be a mapping.")
    return value


def _check_keys(payload: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown keys in {where}: {sorted(unknown)}; allowed: {sorted(allowed)}."
        )


def _path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return Path(os.path.normpath(path if path.is_absolute() else base / path))
