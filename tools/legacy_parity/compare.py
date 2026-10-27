"""Run shortcut-catcher on a legacy dataset directory and compare with a legacy run.

Reads the legacy results written by ``run_legacy.py`` and runs the new tabular
backend with the matching settings: AutoGluon screening, Random Forest seed 0,
candidate choice by verification accuracy, final round by verification macro-F1.
Exits with status 1 if the removal order differs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from shortcut_catcher.core import FixedRounds, ShortcutCatcher
from shortcut_catcher.tabular import (
    ColumnRemover,
    GiniExplainer,
    RandomForestModel,
    load_csv_splits,
)

LEGACY_FILES = {
    "train": "train_normalized.csv",
    "validation": "val_normalized.csv",
    "test": "test_normalized.csv",
    "verification": "visible_normalized.csv",
    "check": "invisible_normalized.csv",
}


def legacy_rounds(results: Path) -> pd.DataFrame:
    """Return one row per legacy round with the feature removed and the scores."""
    visible = pd.read_csv(results / "leaderboard-True-visible-0.15.csv")
    invisible = pd.read_csv(results / "leaderboard-True-invisible-0.15.csv")
    rows = []
    previous: list[str] = []
    for i in range(len(visible)):
        removed = [f for f in str(visible.loc[i, "feature"]).split("+") if f != "base"]
        added = [f for f in removed if f not in previous]
        rows.append(
            {
                "round": i,
                "removed": added[0] if added else "",
                "verification_accuracy": float(visible.loc[i, "score_test"]),
                "verification_f1_macro": float(visible.loc[i, "f1_macro"]),
                "check_f1_macro": float(invisible.loc[i, "f1_macro"]),
            }
        )
        previous = removed
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--legacy-out", type=Path, required=True, help="run_legacy.py --out.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    legacy = legacy_rounds(args.legacy_out / "results")
    splits = load_csv_splits(
        {name: args.data_dir / file for name, file in LEGACY_FILES.items()}, "class"
    )
    loop = ShortcutCatcher(
        RandomForestModel(screening="autogluon"),
        GiniExplainer(),
        ColumnRemover(),
        metric="accuracy",
        max_rounds=len(legacy) - 1,
        window=args.window,
        stopping_rule=FixedRounds(metric="f1_macro"),
        seed=0,
    )
    start = time.perf_counter()
    result = loop.run(splits)
    elapsed = time.perf_counter() - start
    result.history.write(args.out)

    new = pd.DataFrame(
        {
            "round": r.round,
            "removed": r.selected_feature or "",
            "verification_accuracy": r.score("verification", "accuracy"),
            "verification_f1_macro": r.score("verification", "f1_macro"),
            "check_f1_macro": r.score("check", "f1_macro"),
        }
        for r in result.history
    )
    table = legacy.merge(new, on="round", suffixes=("_legacy", "_new"))
    table["same_removal"] = table.removed_legacy == table.removed_new
    for column in ("verification_accuracy", "verification_f1_macro", "check_f1_macro"):
        table[f"{column}_diff"] = (table[f"{column}_new"] - table[f"{column}_legacy"]).abs()
    score_diff = table.filter(like="_diff").to_numpy().max()

    legacy_selected = int(legacy.verification_f1_macro.idxmax())
    report = {
        "rounds_compared": len(table),
        "same_removal_order": bool(table.same_removal.all()),
        "first_divergence": None
        if table.same_removal.all()
        else int(table.loc[~table.same_removal, "round"].iloc[0]),
        "max_score_difference": float(score_diff),
        "selected_round_legacy": legacy_selected,
        "selected_round_new": result.selected_round,
        "new_runtime_seconds": round(elapsed, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "parity_table.csv", index=False)
    (args.out / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(
            table[
                [
                    "round",
                    "removed_legacy",
                    "removed_new",
                    "same_removal",
                    "verification_accuracy_legacy",
                    "verification_accuracy_new",
                ]
            ].to_string(index=False)
        )
    print(json.dumps(report, indent=2))
    ok = report["same_removal_order"] and score_diff <= args.tolerance
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
