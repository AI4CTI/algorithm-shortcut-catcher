"""Run the legacy ShortcutCatcher (AutoGluon) loop on one dataset directory.

Run this with the Python of an environment that has ``autogluon.tabular==1.4.0``
installed (see ``tools/legacy_parity/README.md``). The legacy code is imported
from ``--legacy-dir`` and is not modified; no bytecode is written there.

The dataset directory must contain ``train_normalized.csv``,
``val_normalized.csv``, ``test_normalized.csv``, ``visible_normalized.csv``
and ``invisible_normalized.csv`` with a ``class`` label column, as in the
legacy repository.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--legacy-dir", type=Path, required=True, help="Legacy repository root.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--rounds", type=int, default=10, help="Legacy --rounds (incl. base).")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--max-memory-usage-ratio",
        type=float,
        default=100.0,
        help=(
            "AutoGluon max_memory_usage_ratio for the Random Forest. AutoGluon reduces the "
            "number of trees below 300 when it estimates memory is short, which makes results "
            "depend on free RAM. The default (100) disables that reduction. "
            "Use 1.0 for AutoGluon's own default."
        ),
    )
    args = parser.parse_args()

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.legacy_dir.resolve()))
    import pandas as pd  # noqa: E402
    from loop import Loop  # type: ignore[import-not-found]  # noqa: E402

    data = args.data_dir
    results_path = args.out / "results"
    model_path = args.out / "models"
    results_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    # Same construction as the legacy main.py with --description soft_window,
    # --model_name RandomForestGini, --noise_type removal, --min_bound 0.0,
    # --importance_type default.
    loop = Loop(
        model_name="RandomForestGini",
        noise_type="removal",
        standard=0.15,
        importance_type="default",
        min_bound=0.0,
    )
    rf = loop.model.hyperparameters["RF"][0]
    rf["ag_args_fit"] = {"max_memory_usage_ratio": args.max_memory_usage_ratio}

    loop.set_data(
        train_path=f"{data}/train_normalized.csv",
        val_path=f"{data}/val_normalized.csv",
        test_path=f"{data}/test_normalized.csv",
    )
    start = time.perf_counter()
    loop.iterate_window(
        model_path=str(model_path),
        results_path=str(results_path),
        iid_data=pd.read_csv(f"{data}/test_normalized.csv"),
        visible_ood=pd.read_csv(f"{data}/visible_normalized.csv"),
        invisible_ood=pd.read_csv(f"{data}/invisible_normalized.csv"),
        num_gpus=0,
        rounds=args.rounds,
        window=args.window,
        mode="soft",
    )
    print(f"[legacy] finished in {time.perf_counter() - start:.1f}s; results in {results_path}")


if __name__ == "__main__":
    main()
