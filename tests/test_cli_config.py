import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from shortcut_catcher.backends import build_components, load_splits, make_loop
from shortcut_catcher.cli import app
from shortcut_catcher.config import ConfigError, load_config, parse_config
from shortcut_catcher.core import Patience
from shortcut_catcher.demos.runner import run_demo
from shortcut_catcher.demos.tabular import make_shortcut_splits

runner = CliRunner()


@pytest.fixture
def tabular_files(tmp_path):
    for name, split in make_shortcut_splits(seed=0).items():
        split.X.assign(label=split.y).to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


def write_config(directory, **overrides):
    config = {
        "backend": "tabular",
        "data": {
            "label_column": "label",
            "splits": {n: f"{n}.csv" for n in ("train", "validation", "verification", "check")},
        },
        "loop": {"max_rounds": 2, "window": 3, "seed": 0},
        "model": {"n_estimators": 30, "n_jobs": 1},
    }
    config.update(overrides)
    path = directory / "run.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config))
    return path


def test_parse_config_resolves_paths_and_defaults(tmp_path):
    cfg = parse_config(
        {"backend": "tabular", "data": {"splits": {"train": "a.csv", "verification": "b.csv"}}},
        base_dir=tmp_path,
    )
    assert cfg.data.splits["train"] == tmp_path / "a.csv"
    assert cfg.loop.metric == "f1_macro"
    assert cfg.loop.window == 5


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"data": {"file": "x.csv"}}, "backend"),
        ({"backend": "tabular"}, "data"),
        ({"backend": "tabular", "data": {"file": "x", "splits": {"train": "y"}}}, "either"),
        ({"backend": "tabular", "data": {"file": "x"}, "lop": {}}, "Unknown keys"),
        ({"backend": "tabular", "data": {"splits": {"holdout": "x"}}}, "Unknown split"),
        (
            {"backend": "tabular", "data": {"file": "x"}, "loop": {"stopping": "patience"}},
            "patience",
        ),
    ],
)
def test_parse_config_errors(payload, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(payload)


def test_toml_config(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('backend = "text"\n[data]\nfile = "all.csv"\n[loop]\nmax_rounds = 3\n')
    cfg = load_config(path)
    assert cfg.backend == "text"
    assert cfg.loop.max_rounds == 3


def test_single_file_with_split_column(tmp_path):
    frames = []
    for name, split in make_shortcut_splits(seed=0).items():
        frames.append(split.X.assign(label=split.y, split=name))
    pd.concat(frames).to_csv(tmp_path / "all.csv", index=False)
    cfg = parse_config(
        {"backend": "tabular", "data": {"file": "all.csv", "label_column": "label"}}, tmp_path
    )
    splits = load_splits("tabular", cfg.data)
    assert set(splits) == {"train", "validation", "test", "verification", "check"}
    assert "split" not in splits["train"].features


def test_patience_rule_and_bad_backend_options(tmp_path):
    cfg = parse_config(
        {
            "backend": "tabular",
            "data": {"file": "x.csv"},
            "loop": {"stopping": "patience", "patience": 2},
        }
    )
    loop = make_loop(cfg.loop, build_components("tabular", {}, {}))
    assert isinstance(loop.stopping_rule, Patience)
    with pytest.raises(ConfigError, match="Invalid"):
        build_components("tabular", {"no_such_option": 1}, {})
    with pytest.raises(ConfigError, match="Unknown backend"):
        build_components("images", {}, {})


def test_cli_run_writes_outputs(tabular_files):
    config = write_config(tabular_files)
    out = tabular_files / "out"
    result = runner.invoke(
        app, ["run", "--config", str(config), "--output-dir", str(out), "--json"]
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["backend"] == "tabular"
    assert summary["removed"]
    assert set(summary["kept_features"]).isdisjoint(summary["removed"])
    for name in ("history.json", "rounds.csv", "candidates.csv", "selected.json", "config.json"):
        assert (out / name).is_file()


def test_cli_run_reports_config_errors(tmp_path):
    config = write_config(tmp_path)  # data files do not exist
    result = runner.invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 2
    assert "not found" in result.output


def test_cli_list_backends_json():
    result = runner.invoke(app, ["list-backends", "--json"])
    assert result.exit_code == 0
    rows = {row["name"]: row for row in json.loads(result.stdout)}
    assert rows["tabular"]["installed"] is True
    assert rows["text"]["experimental"] is True


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "shortcut-catcher" in result.stdout


def test_tabular_demo_finds_all_planted_shortcuts():
    outcome = run_demo("tabular")
    assert outcome.found == outcome.shortcuts
    assert outcome.other_removed == ()
    assert outcome.selected_score > outcome.baseline_score + 0.1
    # A genuine feature outranks every shortcut before round 1.
    assert min(r for r in outcome.ranks_before_round_1().values() if r is not None) > 1


def test_cli_demo_json(tmp_path):
    result = runner.invoke(app, ["demo", "tabular", "-o", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["seed"] == 0
    assert sorted(summary["shortcuts_found"]) == sorted(summary["planted_shortcuts"])
    assert summary["other_removed"] == []


@pytest.mark.slow
def test_text_demo_finds_all_planted_shortcuts():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    outcome = run_demo("text")  # downloads ELECTRA-small on first use
    assert outcome.found == outcome.shortcuts
    assert outcome.other_removed == ()
    assert outcome.selected_score > outcome.baseline_score + 0.3


def test_cli_messages_keep_square_brackets(monkeypatch):
    # Rich markup must not swallow "[text]" in install hints or bracketed feature names.
    from shortcut_catcher import backends

    info = backends.BACKENDS["text"]
    monkeypatch.setitem(
        backends.BACKENDS,
        "text",
        backends.BackendInfo(
            info.name, info.description, info.experimental, ("no_such_module",), info.install
        ),
    )
    result = runner.invoke(app, ["demo", "text"])
    assert result.exit_code == 2
    assert ".[text]" in result.output
    listing = runner.invoke(app, ["list-backends"])
    assert "[text]" in listing.output
