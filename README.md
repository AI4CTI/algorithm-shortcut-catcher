# shortcut-catcher

[![CI](https://github.com/AI4CTI/shortcut-catcher/actions/workflows/ci.yml/badge.svg)](https://github.com/AI4CTI/shortcut-catcher/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![Paper](https://img.shields.io/badge/DOI-10.1145%2F3808671-blue.svg)](https://doi.org/10.1145/3808671)

`shortcut-catcher` finds and removes **shortcut features**: features a model exploits on its training data that do not hold where the model is deployed. It ranks the features the model relies on with an explainer, retrains without each top-ranked candidate, and keeps the removal that scores best on an independently collected **verification set**. Removals accumulate over rounds, and the round with the best verification score is selected.

This is the reference implementation of **ShortcutCatcher**, published in *Proceedings of the ACM on Networking* (CoNEXT 2026) for encrypted traffic classification. The algorithm does not depend on the data type: the package ships a tabular backend (Random Forest with Gini importance, used in the paper) and an **experimental** text backend (frozen ELECTRA encoder).

> **Citation.** If you use this tool in academic work, please cite the paper; see [Citation](#citation) and [`CITATION.cff`](CITATION.cff).

> **Funding.** This work was carried out within **AI4CTI**, a joint research initiative between [Politecnico di Torino](https://www.polito.it/) and [Ermes Browser Security](https://www.ermes.company/), funded by the **Italian Ministry of Education, Grant FISA-2023-00168**.

## Contents

- [Background](#background)
- [How it works](#how-it-works)
- [Install](#install)
- [Quickstart](#quickstart)
- [Running on your own data](#running-on-your-own-data)
- [CLI reference](#cli-reference)
- [Python API](#python-api)
- [Adding a backend](#adding-a-backend)
- [Reproducing the paper](#reproducing-the-paper)
- [Limitations](#limitations)
- [Development](#development)
- [Citation](#citation) · [Funding](#funding) · [Licence](#licence) · [Acknowledgements](#acknowledgements)

## Background

A classifier can reach high accuracy on its test set by relying on features that are artefacts of how the training data was collected: an IP address, a checksum, a timestamp, a word that happens to appear in one class. These **shortcuts** are predictive on data from the same collection, so a held-out test split does not reveal them. They fail once the model meets data from a different scenario.

ShortcutCatcher uses two datasets with the same features: the training data, and a smaller **verification set** collected in a different scenario. A shortcut helps on the first and hurts on the second. The loop removes, one at a time, the features whose removal improves the verification score.

**How this differs from recursive feature elimination (RFE).** RFE removes the features a model *ignores*, to make it smaller while keeping accuracy on the same distribution. ShortcutCatcher removes the features a model *exploits* but that fail to generalise, and judges each removal on an independent verification set rather than on the training distribution. Its candidates are the most important features, not the least important ones.

## How it works

```text
removed = []
model   = fit(train)
for round in 1..max_rounds:
    ranking    = explainer(model, train)              # most important features first
    candidates = top `window` features not yet removed
    for each candidate c:                             # one retraining per candidate
        model_c = fit(train without removed + [c])
        score_c = verification score of model_c
    winner   = candidate with the best score_c        # ties keep the higher-ranked one
    removed += [winner];  model = model_winner
selected = round with the best verification score      # round 0 (nothing removed) included
```

Each round costs `window` model fits. The winner of a round is adopted even if every candidate lowers the verification score; the final selection step then picks the best round. With `max_window` set, a round first evaluates `window` candidates and keeps adding candidates while none of them improves on the previous round.

The data can have up to five splits:

| Split | Role | Required |
| --- | --- | --- |
| `train` | Fits the model; explained to rank features. | yes |
| `validation` | Picks the best training epoch (text) and takes part in column screening (tabular). | text: yes; tabular: no |
| `test` | Same scenario as `train`; reported only. | no |
| `verification` | Different scenario; picks the winner of each round and the final round. | yes |
| `check` | Different scenario, never used for decisions: a fair estimate for the selected model. | no, but recommended |

Backends:

| Component | `tabular` | `text` (experimental) |
| --- | --- | --- |
| Model | scikit-learn Random Forest, settings of the published experiments | frozen ELECTRA encoder + gated attention pooling + MLP head |
| Explainer | Gini (impurity) importance | pooling weights → word scores → per-sample log-MAD outliers → Count or V8 ranking |
| Remover | drop columns | delete words (case-insensitive, whole words) |

## Install

Python 3.10 to 3.13. The package is not yet on PyPI; install it from source.

```bash
git clone https://github.com/AI4CTI/shortcut-catcher.git
cd shortcut-catcher

# Core and tabular backend
uv sync                     # or: pip install .

# With the experimental text backend (PyTorch + Transformers)
uv sync --extra text        # or: pip install ".[text]"
```

With `uv` on Linux, the text extra installs the CPU build of PyTorch. For a GPU, install PyTorch first following the [official instructions](https://pytorch.org/get-started/locally/), then `pip install ".[text]"`.

The text backend downloads its encoder from Hugging Face on first use (`google/electra-small-discriminator` by default, about 55 MB). Set `HF_HOME` to choose the cache directory.

Check what is installed:

```bash
uv run shortcut-catcher list-backends
```

## Quickstart

Both demos generate their data inside the repository: no dataset is shipped or downloaded. Each plants **three shortcuts of different strength** with ordinary-looking names, so neither the names nor the explainer ranking alone give them away. They run on a laptop CPU in under a minute, with fixed seeds.

**Tabular.** A four-class, flow-like dataset (`video`, `web`, `chat`, `voip`) with five genuine features of different strength (`pkt_len_mean`, `iat_mean`, ...), three noise columns, one constant column and three planted shortcuts:

| Shortcut | In training | At deployment (verification/check) |
| --- | --- | --- |
| `ttl_initial` (strong) | `web` flows mostly captured on a host with initial TTL 128, the rest with TTL 64 | TTL unrelated to the class |
| `capture_hour` (moderate) | `chat` flows mostly captured at night | every class at every hour |
| `ip_id_jitter` (weak) | higher for `video` | higher for `voip`: it now misleads |

```bash
uv run shortcut-catcher demo tabular
```

With the default seed, a genuine feature (`iat_mean`) is ranked first; the shortcuts are ranked 2nd to 4th. The loop removes `ip_id_jitter` first, because that removal helps the verification score most, then `ttl_initial` and `capture_hour`. Rounds 4–6 remove other features and no longer help, so round 3 is selected: exactly the three shortcuts, with verification macro-F1 from about 0.50 to 0.67 and check macro-F1 from about 0.46 to 0.67.

**Text (experimental).** Two-clause movie reviews labelled `pos` or `neg`, with sentiment adjectives as the real signal, and three ordinary words planted at different rates:

| Shortcut | In training | At deployment |
| --- | --- | --- |
| `really` (strong) | "really" before an adjective in most positive reviews | in most negative reviews |
| `honestly` (moderate) | clause ending in many negative reviews | in many positive reviews |
| `overall` (weak) | somewhat more frequent in positive reviews | somewhat more frequent in negative reviews |

```bash
uv run --extra text shortcut-catcher demo text
```

With the default seed, `really` and `honestly` are ranked 1st and 2nd, `overall` only 12th. The loop removes `really`, then `overall`, then `honestly`; verification macro-F1 goes from about 0.39 to 0.83 and check macro-F1 from about 0.42 to 0.84. The first run downloads ELECTRA-small.

The demo output ends with a table of the planted shortcuts: their rank before round 1, the round in which each was removed, and whether it is in the selected set. With other seeds (`--seed N`) the outcome varies, especially for the text demo; see [Limitations](#limitations). The same demos as plain scripts: [`examples/tabular_demo.py`](examples/tabular_demo.py), [`examples/text_demo.py`](examples/text_demo.py).

## Running on your own data

Describe the run in a YAML or TOML file and pass it to `shortcut-catcher run`. Relative paths are resolved against the configuration file.

```yaml
backend: tabular            # or: text
data:
  label_column: class
  splits:                   # one CSV/Parquet per split ...
    train: data/train.csv
    validation: data/val.csv
    verification: data/verification.csv
    check: data/check.csv
  # ... or one file with a split column:
  # file: data/all.csv
  # split_column: split
  # text_column: text       # text backend
loop:
  metric: f1_macro          # tabular: accuracy, f1_macro, f1_micro, precision_macro, recall_macro, nll; text: accuracy, f1_macro, nll
  max_rounds: 10
  window: 5
  max_window: null          # set to extend the window while no candidate helps
  extension_step: 1
  seed: 0
  stopping: fixed           # or: patience (then set `patience: N`)
  selection_metric: null    # metric for the final round choice; defaults to `metric`
model: {}                   # backend model options, see below
explainer: {}               # backend explainer options (text only)
output_dir: runs/my-run
```

```bash
uv run shortcut-catcher run --config run.yaml
```

To try this on files, `uv run python examples/make_example_data.py examples/data` writes the demo data as CSV; [`examples/tabular.yaml`](examples/tabular.yaml) and [`examples/text.toml`](examples/text.toml) run on it.

**Tabular data.** Numeric feature columns plus a label column; encode categorical features beforehand. `model` options are passed to `RandomForestModel`: any `RandomForestClassifier` parameter (defaults: `n_estimators=300`, `max_leaf_nodes=15000`, `criterion="gini"`, `bootstrap=True`, `n_jobs=-1`) and `screening`:

- `autogluon` (default) reproduces the column preprocessing of AutoGluon 1.4, which the paper's experiments used: columns are screened on train and validation together, constant columns are dropped, two-valued columns become 0/1 indicators, and duplicate columns are dropped. AutoGluon detects duplicates on a 3000-row sample, so a column that matches another one on the sample but not on all rows is also dropped.
- `exact` drops constant columns and exact duplicates only, checked on all training rows.

**Text data.** A text column and a label column. `model` options go to `ElectraModel` (`checkpoint`, `head`, `epochs`, `batch_size`, `learning_rate`, `checkpoint_metric`, `early_stopping_patience`, `max_length`, `device`, ...); `explainer` options go to `PoolingExplainer` (`ranking: count | v8`, `token_to_word`, `outlier_z`, `outlier_min_ratio`, `outlier_min_words`, `min_df`, `max_entropy`, `min_outliers`, ...). See the module docstrings for details.

**Outputs** in `output_dir`:

| File | Contents |
| --- | --- |
| `history.json` | Every round: removed features, metrics on every split, evaluated candidates, explainer ranking head |
| `rounds.csv` | One row per round with metrics per split |
| `candidates.csv` | One row per evaluated candidate: rank, explainer score, verification score, gain |
| `selected.json` | Selected round, removed features, kept features (tabular), metrics |
| `config.json` | The configuration of the run |

## CLI reference

```text
shortcut-catcher [--version] [--verbose] COMMAND

  run            --config PATH [--output-dir PATH] [--json]
                 Run the loop on your own data, as described by a configuration file.
  demo           {tabular|text} [--output-dir PATH] [--seed INT] [--json]
                 Run a bundled demo on synthetic data with several planted shortcuts.
  list-backends  [--json]
                 Show the available backends and whether their dependencies are installed.
```

- `--json` prints a machine-readable summary on stdout; progress goes to stderr.
- `--verbose` also prints every evaluated candidate and debug logs.
- Without `--output-dir`, results go to `runs/<command>-<timestamp>/`, or under `$SHORTCUT_CATCHER_OUTPUT_DIR` if set. A `.env` file in the working directory is read; see [`.env.example`](.env.example).

## Python API

```python
from shortcut_catcher import ShortcutCatcher
from shortcut_catcher.tabular import ColumnRemover, GiniExplainer, RandomForestModel, load_csv_splits

splits = load_csv_splits(
    {"train": "train.csv", "validation": "val.csv", "verification": "verification.csv", "check": "check.csv"},
    label_column="class",
)
loop = ShortcutCatcher(RandomForestModel(), GiniExplainer(), ColumnRemover(), metric="f1_macro", max_rounds=10, window=5, seed=0)
result = loop.run(splits)

print(result.selected_round, result.removed)
result.history.write("runs/my-run")
```

`result.model` is the fitted model of the selected round; `result.history` holds every round.

## Adding a backend

The loop in `shortcut_catcher.core` only talks to four components, defined as `typing.Protocol` classes in [`core/interfaces.py`](src/shortcut_catcher/core/interfaces.py). The dataset type is yours: the core passes it between your components without looking inside.

```python
class MyModel:                                    # Model
    def fit(self, train, validation, *, seed):    # train a fresh model, return a fitted model
        ...

class MyFitted:                                   # FittedModel (returned by fit)
    def evaluate(self, data) -> dict[str, float]: # e.g. {"accuracy": ..., "f1_macro": ...}
        ...

class MyExplainer:                                # Explainer
    def rank(self, model, data) -> list[tuple[str, float]]:  # eligible features, most important first
        ...

class MyRemover:                                  # Remover
    def remove(self, data, features):             # return a copy of data without `features`
        ...

ShortcutCatcher(MyModel(), MyExplainer(), MyRemover(), metric="accuracy").run(
    {"train": ..., "validation": ..., "verification": ...}
)
```

Requirements: `fit` must not reuse state from earlier fits (it is called once per candidate), must use the given `seed`, and `evaluate` must return the metric the loop selects on. Features left out of the ranking are never candidates. A `StoppingRule` (`should_stop(history)`, `select(history)`) changes when the loop stops and which round is selected; `FixedRounds` and `Patience` are provided. The test suite has a complete toy backend in [`tests/core/fakes.py`](tests/core/fakes.py).

## Reproducing the paper

The datasets and the original code of the paper are in [SmartData-Polito/ShortcutCatcher](https://github.com/SmartData-Polito/ShortcutCatcher). With the settings of the original runs (`screening="autogluon"`, which is the default, seed 0, `metric="accuracy"` for each round and the final round chosen by macro-F1), the tabular backend reproduces the original loop exactly: on the `app53-time-s0` dataset, ten rounds with window 5 remove the same features in the same order, with the same scores.

[`tools/legacy_parity/`](tools/legacy_parity/README.md) has the scripts and step-by-step instructions to run this check yourself, including the AutoGluon environment the original code needs. It also documents a property of the original pipeline: AutoGluon trains fewer than 300 trees when it estimates memory is short, so the original results can vary with the machine.

The MLP and TabICL results of the paper used AutoGluon models with permutation importance; they are not part of this release.

## Limitations

- **The verification set is used for every decision.** It picks the winner of every round and the final round, so the selected verification score is optimistic. Report the `check` split, which takes no part in any decision. A small verification set can also make the loop remove a genuine feature because it raises the verification score by chance.
- **Greedy, one feature per round.** Each round adds one feature to the removal set and never puts one back. Shortcuts interact: a weak shortcut may only become visible after a stronger one is removed, and while a shortcut is still present, removing a genuine feature that counteracts its bias can raise the verification score more than removing the shortcut itself. A larger `window` evaluates more candidates per round but also gives noise more chances.
- **Shortcuts the model learns but the verification set does not punish are not removed.** A feature that stops being informative at deployment, rather than misleading, costs little on the verification set, and its removal may not stand out from noise.
- **Cost.** Each round trains `window` models; a run trains about `1 + max_rounds × window` models.
- **Classification only**, with metrics computed from predicted labels and probabilities.
- **Tabular backend:** numeric features only. Gini importance is computed on the training data and favours continuous, high-cardinality features. The default `autogluon` screening can drop a column that is not an exact duplicate (see above); use `screening: exact` if you do not need to reproduce the paper.
- **Text backend (experimental):** a port of research code that has not been validated beyond the original experiments. Pooling weights show where the model attends, not a causal effect of a word. Only whole words are removed and punctuation is kept, so an artifact such as a `(source)` tag or a `Prefix:` leaves `()` or `:` behind, which the model can keep using and the word-level explainer cannot rank; texts are re-tokenised afterwards. Encoder outputs are cached in memory for every distinct text, which grows with corpus size × sequence length × hidden size. The default encoder is ELECTRA-small, chosen to run on a CPU; the original experiments used ELECTRA-base on GPU.

## Development

```bash
uv sync --extra dev --extra text
uv run pre-commit install
uv run pytest                    # add -m "not slow" to skip the real-model text demo test
uv run ruff check . && uv run ruff format --check .
uv run mypy                      # strict, on src/shortcut_catcher/core
```

Tests run on CPU. Text tests use a tiny offline encoder and are skipped when the `[text]` extra is not installed. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

```bibtex
@article{zhao2026shortcutcatcher,
  title     = {ShortcutCatcher: Making Traffic Classification Reliable},
  author    = {Zhao, Yuqi and Boffa, Matteo and Vassio, Luca and Mellia, Marco},
  journal   = {Proceedings of the ACM on Networking},
  volume    = {4},
  number    = {CoNEXT2},
  pages     = {1--15},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3808671}
}
```

## Funding

This work was carried out within **AI4CTI**, a joint research initiative between Politecnico di Torino and Ermes Browser Security, funded by the Italian Ministry of Education, Grant FISA-2023-00168.

## Licence

[GNU Affero General Public License v3.0 or later](LICENSE).

## Acknowledgements

- The tabular backend ports the implementation used for the paper, [SmartData-Polito/ShortcutCatcher](https://github.com/SmartData-Polito/ShortcutCatcher), which runs on [AutoGluon](https://github.com/autogluon/autogluon); the `autogluon` screening mode re-implements AutoGluon's preprocessing for numeric data.
- The text backend uses [ELECTRA](https://arxiv.org/abs/2003.10555) (Clark et al., ICLR 2020) through [Hugging Face Transformers](https://github.com/huggingface/transformers), with gated attention pooling from Ilse et al. (ICML 2018).
- Built with [scikit-learn](https://scikit-learn.org), [PyTorch](https://pytorch.org), [Typer](https://typer.tiangolo.com) and [Rich](https://github.com/Textualize/rich).
