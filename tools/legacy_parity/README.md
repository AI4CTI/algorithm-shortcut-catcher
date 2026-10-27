# Parity check against the legacy implementation

These two scripts check that the tabular backend selects the same features as the
original ShortcutCatcher code used for the CoNEXT 2026 paper, on the same data.
They are not needed to use the package.

- `run_legacy.py` runs the legacy loop (AutoGluon 1.4 Random Forest). The legacy
  code is imported from its own directory and is not modified.
- `compare.py` runs `shortcut-catcher` with the matching settings and compares
  the two runs round by round.

## What you need

| Item | Notes |
| --- | --- |
| A checkout of the legacy code | [SmartData-Polito/ShortcutCatcher](https://github.com/SmartData-Polito/ShortcutCatcher): the directory that contains `loop.py`, `model.py`, `explain.py`, `noise.py`. |
| A legacy-format dataset directory (the legacy repository ships the paper's datasets under `dataset/`) | `train_normalized.csv`, `val_normalized.csv`, `test_normalized.csv`, `visible_normalized.csv`, `invisible_normalized.csv`, label column `class`, numeric features. |
| [`uv`](https://docs.astral.sh/uv/) | Used to create both environments. |
| Python 3.10 | For the legacy environment (the legacy code was developed on 3.10). |
| Disk and network | The legacy environment takes about 0.4 GB. |
| Time | About 1 minute per legacy round on a laptop for a 15k-row dataset with window 5; the new code takes about the same. |

## 1. Legacy environment (AutoGluon)

```bash
uv venv -p 3.10 legacy-ag-env
VIRTUAL_ENV=legacy-ag-env uv pip install "autogluon.tabular==1.4.0" typing_extensions
# Same library versions as the comparison below; lets compare.py run here too.
VIRTUAL_ENV=legacy-ag-env uv pip install --no-deps -e /path/to/shortcut-catcher
```

`typing_extensions` is needed because `autogluon.core` 1.4.0 imports it without
declaring it as a dependency. The legacy README installs the full `autogluon==1.4`
meta-package; the legacy code only imports `autogluon.tabular`, so the tabular
package is enough. We tested with `autogluon.tabular 1.4.0`, `scikit-learn 1.7.2`,
`pandas 2.3.3`, `numpy 2.2.6`.

## 2. Run the legacy loop

```bash
PYTHONDONTWRITEBYTECODE=1 legacy-ag-env/bin/python tools/legacy_parity/run_legacy.py \
  --legacy-dir /path/to/legacy/ShortcutCatcher \
  --data-dir /path/to/dataset/per-flow/app53-time-s0 \
  --out runs/parity/legacy \
  --rounds 10 --window 5
```

`--rounds` counts the baseline, as in the legacy `main.py`: 10 rounds means the
baseline plus 9 removals.

## 3. Run the new code and compare

```bash
legacy-ag-env/bin/python tools/legacy_parity/compare.py \
  --data-dir /path/to/dataset/per-flow/app53-time-s0 \
  --legacy-out runs/parity/legacy \
  --out runs/parity/new --window 5
```

`compare.py` prints the removal order of both runs and writes
`parity_table.csv`, `parity_report.json` and the run history to `--out`. It exits
with status 1 if the removal order or any score differs.

Running `compare.py` from the package's own environment (`uv run python
tools/legacy_parity/compare.py ...`) uses newer library versions; see the results below.

## Settings that make the runs comparable

| Setting | Legacy | shortcut-catcher |
| --- | --- | --- |
| Model | AutoGluon `RF` with `criterion=gini` | `RandomForestModel()`; same forest parameters, `seed=0` (AutoGluon's fixed `random_state`) |
| Column preprocessing | AutoGluon feature pipeline | `screening="autogluon"`, a re-implementation of that pipeline for numeric data |
| Explainer | `feature_importances_` | `GiniExplainer()` |
| Removal | column set to 0 in train/validation, then dropped by AutoGluon as constant | column dropped from every split |
| Candidate choice | best verification (`visible`) accuracy among the top 5 | `metric="accuracy"`, `window=5` |
| Final round | best verification macro-F1 (`evaluate/table2.py`) | `FixedRounds(metric="f1_macro")` |

## Behaviour of the legacy pipeline worth knowing

1. **Number of trees depends on free memory.** AutoGluon trains fewer than 300 trees
   when it estimates the forest would use more than 15% of available memory. On a
   16 GB laptop the 15k-row, 27-class example trained 213 trees instead of 300.
   Results therefore change with the machine and its load. `run_legacy.py` lifts
   that limit by default (`--max-memory-usage-ratio 100`), so both runs train 300
   trees; pass `--max-memory-usage-ratio 1` to get AutoGluon's own behaviour.
2. **Columns are screened on train and validation together.** A column constant in
   train but not in validation is kept.
3. **Duplicate detection uses a row sample.** Two columns are treated as duplicates
   if they agree on a sample of 3000 rows drawn with replacement, even if they differ
   elsewhere. In `app53-time-s0`, `ip_ttl_pkt3` is dropped this way: it equals
   `ip_ttl_pkt0` on the sample but not on all rows. `screening="exact"` checks all
   rows instead.
4. **Two-valued columns become 0/1 indicators**; values not seen in training map to 0.

## Results

Checked on 2026-10-27 on the `app53-time-s0` per-flow dataset of the paper, with
`--rounds 10 --window 5` (baseline plus 9 removals), on a laptop with 8 cores and 16 GB of RAM:

| shortcut-catcher environment | Same removal order | Max score difference | Same selected round |
| --- | --- | --- | --- |
| legacy environment (scikit-learn 1.7.2) | yes, 9/9 | < 1e-16 | yes |
| package environment (scikit-learn 1.9.1) | yes, 9/9 | 0 | yes |

Runtime: legacy about 7 minutes, shortcut-catcher about 4–5 minutes.
