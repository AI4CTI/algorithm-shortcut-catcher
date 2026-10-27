# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-10-30

First public release under the [AI4CTI](https://github.com/AI4CTI) organisation. It merges the code of the CoNEXT 2026 paper (tabular, traffic classification) and a text-classification extension into one package where the algorithm does not depend on the data type. The history of the original research repositories is not part of this repository.

### Added
- Application-agnostic core: `ShortcutCatcher` loop, `Model` / `Explainer` / `Remover` / `StoppingRule` protocols, per-round candidate search with optional adaptive window, `FixedRounds` and `Patience` rules, and a `History` with JSON/CSV export.
- Tabular backend: `RandomForestModel` (scikit-learn, settings of the published AutoGluon 1.4 runs), `GiniExplainer`, `ColumnRemover`, `TabularData` and `load_csv_splits`.
- Column screening before each fit: `"autogluon"` reproduces the AutoGluon 1.4 feature pipeline used by the published experiments; `"exact"` drops only constant columns and exact duplicates.
- Experimental text backend behind the `[text]` extra: frozen ELECTRA encoder with a gated (or plain) attention-pooling head (`ElectraModel`), pooling-weight word ranking with per-sample log-MAD outliers and Count or V8 global ranking (`PoolingExplainer`), whole-word removal (`WordRemover`), `TextData` and `load_text_splits`.
- `tools/legacy_parity/`: scripts and instructions to check the tabular backend against the legacy implementation.
- Typer command-line interface with `run`, `demo` and `list-backends`, Rich output, `--json` summaries, YAML/TOML run configuration and `.env` support.
- Demos on synthetic data with three planted shortcuts of different strength each, for both backends (`shortcut-catcher demo tabular|text`, `examples/`).
- Repository tooling: PEP 621 `pyproject.toml`, `uv` lockfile, `hatchling`, `ruff`, strict `mypy` on the core, `pre-commit`, GitHub Actions CI on Python 3.10 to 3.13 (CPU only), AGPL-3.0-or-later licence, `CITATION.cff`.

### Changed compared to the research code
- The tabular backend uses scikit-learn directly instead of AutoGluon. AutoGluon 1.4 supports Python up to 3.12, pins older numpy/pandas/scikit-learn versions, and reduces the number of trees when it estimates memory is short, which made results depend on the machine. The `autogluon` screening mode keeps the original column preprocessing so results stay identical.
- The text backend defaults to ELECTRA-small on CPU (the research code used ELECTRA-base on GPU with BF16) and caches encoder outputs in memory.

### Not included
- Datasets, trained models and experiment results.
- AutoGluon models other than Random Forest (MLP, TabICL) and permutation importance, used for one table of the paper.
- Text-backend research variants: GPU worker pools and shared-memory caches, BF16, other attribution methods (Input×Gradient, AttnLRP, Integrated Gradients), ranking variants other than Count and V8, the TF-IDF baseline, warm starts and class-weighted training.
