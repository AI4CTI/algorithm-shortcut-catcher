# Contributing to shortcut-catcher

Thank you for your interest in contributing. This is a research tool, so the workflow is lightweight.

## Code of conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). By participating you agree to abide by its terms.

## Development setup

You need [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AI4CTI/shortcut-catcher.git
cd shortcut-catcher
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

## Coding conventions

- **Formatter and linter:** [`ruff`](https://docs.astral.sh/ruff/), configured in `pyproject.toml`. Run `uv run ruff format` and `uv run ruff check --fix`.
- **Type checking:** [`mypy`](https://mypy-lang.org/) in strict mode on `src/shortcut_catcher/core/`. Type hints on all public functions.
- **Tests:** [`pytest`](https://pytest.org), CPU only. Mark long tests with `@pytest.mark.slow`.
- **Seeds:** pass seeds as explicit parameters; never rely on global random state.
- **Data:** the repository ships code only. Do not commit datasets, trained models, result files, internal paths or credentials. Tests and demos generate their data.
- **Commits:** sign your commits with `git commit -s` (DCO sign-off). We do not require a CLA.

CI runs lint, type checks and tests on Python 3.10 to 3.13 for every pull request.

## Core rules

`src/shortcut_catcher/core/` holds the algorithm and must stay independent of any data type: no numpy, pandas, scikit-learn or torch imports. A test enforces this.

## Reporting security vulnerabilities

Please do not file public issues for security-sensitive matters. See [`SECURITY.md`](SECURITY.md).
