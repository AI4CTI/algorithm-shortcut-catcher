"""Command-line interface: ``shortcut-catcher``."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .backends import BACKENDS, config_to_dict, run_config, summarize, write_outputs
from .config import ConfigError, load_config
from .core import CandidateResult, LoopResult, RoundRecord
from .demos.runner import run_demo

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "ShortcutCatcher: find and remove shortcut features, i.e. features a model exploits "
        "on its training data that do not generalise, using an independent verification set. "
        "The text backend is experimental."
    ),
)
out = Console()
err = Console(stderr=True)

OUTPUT_ENV = "SHORTCUT_CATCHER_OUTPUT_DIR"
_state = {"verbose": False}


class DemoName(str, Enum):
    tabular = "tabular"
    text = "text"


def _version(value: bool) -> None:
    if value:
        out.print(f"shortcut-catcher {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version."),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Also show every candidate and debug logs.")
    ] = False,
) -> None:
    _load_dotenv()
    _state["verbose"] = verbose
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(Path.cwd() / ".env", override=False)


def _fmt(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value:.4f}"


def _progress(enabled: bool, metric: str) -> tuple[Any, Any]:
    def on_round(record: RoundRecord) -> None:
        if not enabled:
            return
        verification = _fmt(record.score("verification", metric))
        if record.round == 0:
            err.print(f"[bold]Baseline[/bold]: verification {metric} {verification}")
        else:
            err.print(
                f"[bold]Round {record.round}[/bold]: removed [cyan]{escape(str(record.selected_feature))}[/cyan], "
                f"verification {metric} {verification}"
            )

    def on_candidate(round_index: int, candidate: CandidateResult) -> None:
        if enabled and _state["verbose"]:
            err.print(
                f"  round {round_index} candidate {candidate.rank} [dim]{escape(candidate.feature)}[/dim]: "
                f"{metric} {_fmt(candidate.score)} (gain {candidate.gain:+.4f})",
                highlight=False,
            )

    return on_round, on_candidate


def _table(result: LoopResult, metric: str) -> Table:
    has_check = any("check" in r.metrics for r in result.history)
    table = Table(title=f"Rounds ({metric})")
    table.add_column("round", justify="right")
    table.add_column("removed")
    table.add_column("verification", justify="right")
    if has_check:
        table.add_column("check", justify="right")
    for record in result.history:
        row = [
            str(record.round),
            escape(record.selected_feature) if record.selected_feature else "(baseline)",
            _fmt(record.score("verification", metric)),
        ]
        if has_check:
            row.append(_fmt(record.score("check", metric)))
        style = "bold green" if record.round == result.selected_round else None
        table.add_row(*row, style=style)
    table.caption = f"Selected round {result.selected_round} (highlighted)."
    return table


def _output_dir(option: Path | None, configured: Path | None, default_name: str) -> Path:
    if option is not None:
        return option
    if configured is not None:
        return configured
    root = Path(os.environ.get(OUTPUT_ENV, "runs"))
    return root / f"{default_name}-{time.strftime('%Y%m%d-%H%M%S')}"


@app.command()
def run(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Run configuration (.yaml or .toml).")
    ],
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Where to write the results.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print a JSON summary to stdout.")
    ] = False,
) -> None:
    """Run the loop on your own data, as described by a configuration file."""
    try:
        cfg = load_config(config)
        on_round, on_candidate = _progress(not as_json, cfg.loop.metric)
        result = run_config(cfg, on_round=on_round, on_candidate=on_candidate)
    except (ConfigError, ValueError, KeyError) as error:
        err.print(f"[red]Error:[/red] {escape(str(error))}")
        raise typer.Exit(2) from error
    directory = _output_dir(output_dir, cfg.output_dir, f"run-{cfg.backend}")
    paths = write_outputs(result, directory, metric=cfg.loop.metric, config=config_to_dict(cfg))
    _report(result, cfg.loop.metric, directory, paths, as_json, extra={"backend": cfg.backend})


@app.command()
def demo(
    name: Annotated[DemoName, typer.Argument(help="Which demo to run.")],
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Where to write the results.")
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            help="Random seed for data and models [default: the seed of the README example]."
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print a JSON summary to stdout.")
    ] = False,
) -> None:
    """Run a bundled demo on synthetic data with several planted shortcuts.

    The text demo is experimental and downloads ELECTRA-small (about 55 MB) from
    Hugging Face on first use.
    """
    if not BACKENDS[name.value].available:
        err.print(
            f"[red]Error:[/red] the {name.value} demo needs: {escape(BACKENDS[name.value].install)}"
        )
        raise typer.Exit(2)
    on_round, on_candidate = _progress(not as_json, "f1_macro")
    if not as_json:
        err.print(f"Running the {name.value} demo...")
    outcome = run_demo(name.value, seed=seed, on_round=on_round, on_candidate=on_candidate)
    directory = _output_dir(output_dir, None, f"demo-{name.value}")
    paths = write_outputs(
        outcome.result,
        directory,
        metric=outcome.metric,
        config={"demo": name.value, "seed": outcome.seed},
    )
    extra = {
        "demo": name.value,
        "seed": outcome.seed,
        "planted_shortcuts": list(outcome.shortcuts),
        "shortcut_ranks_before_round_1": outcome.ranks_before_round_1(),
        "shortcut_removal_rounds": outcome.removal_rounds,
        "shortcuts_found": list(outcome.found),
        "shortcuts_missed": list(outcome.missed),
        "other_removed": list(outcome.other_removed),
    }
    _report(outcome.result, outcome.metric, directory, paths, as_json, extra=extra)
    if as_json:
        return
    table = Table(title=f"Planted shortcuts (seed {outcome.seed})")
    for column in ("shortcut", "rank before round 1", "removed in round", "in selected set"):
        table.add_column(column)
    ranks, rounds = outcome.ranks_before_round_1(), outcome.removal_rounds
    for shortcut in outcome.shortcuts:
        table.add_row(
            escape(shortcut),
            "-" if ranks[shortcut] is None else str(ranks[shortcut]),
            "-" if rounds[shortcut] is None else str(rounds[shortcut]),
            "[green]yes[/green]" if shortcut in outcome.found else "[yellow]no[/yellow]",
        )
    out.print(table)
    other = escape(", ".join(outcome.other_removed)) or "none"
    out.print(
        f"Found {len(outcome.found)} of {len(outcome.shortcuts)} planted shortcuts; "
        f"other features removed: {other}. Verification {outcome.metric}: "
        f"{_fmt(outcome.baseline_score)} (baseline) -> {_fmt(outcome.selected_score)} (selected round)."
    )


def _report(
    result: LoopResult,
    metric: str,
    directory: Path,
    paths: dict[str, Path],
    as_json: bool,
    *,
    extra: dict[str, Any],
) -> None:
    if as_json:
        payload = {**extra, **summarize(result, metric), "output_dir": str(directory)}
        sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
        return
    out.print(_table(result, metric))
    removed = escape(", ".join(result.removed)) or "(nothing)"
    out.print(f"Removed in the selected round: {removed}")
    out.print(
        f"Results written to [bold]{escape(str(directory))}[/bold] ({', '.join(p.name for p in paths.values())})."
    )


@app.command("list-backends")
def list_backends(
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON to stdout.")] = False,
) -> None:
    """Show the available backends and whether their dependencies are installed."""
    rows = [
        {
            "name": info.name,
            "installed": info.available,
            "experimental": info.experimental,
            "description": info.description,
            "install": info.install,
        }
        for info in BACKENDS.values()
    ]
    if as_json:
        sys.stdout.write(json.dumps(rows, indent=2) + "\n")
        return
    table = Table(title="Backends")
    for column in ("name", "status", "installed", "description"):
        table.add_column(column)
    for row in rows:
        installed = "yes" if row["installed"] else f"[red]no[/red]: {escape(str(row['install']))}"
        table.add_row(
            str(row["name"]),
            "experimental" if row["experimental"] else "stable",
            installed,
            str(row["description"]),
        )
    out.print(table)
