"""Tabular demo: three planted shortcuts of different strength among flow-like features.

Run with:  uv run python examples/tabular_demo.py
Same as:   shortcut-catcher demo tabular
"""

from shortcut_catcher.demos.runner import run_demo


def main() -> None:
    outcome = run_demo("tabular")  # default seed, as in the README
    for record in outcome.result.history:
        print(
            f"round {record.round}: removed {record.selected_feature or '-':<16} "
            f"verification f1_macro {record.score('verification', 'f1_macro'):.4f}  "
            f"check f1_macro {record.score('check', 'f1_macro'):.4f}"
        )
    print("Planted shortcuts, rank before round 1:", outcome.ranks_before_round_1())
    print("Planted shortcuts, removal round:", outcome.removal_rounds)
    print(f"Selected round {outcome.result.selected_round}: removed {list(outcome.result.removed)}")
    print(
        f"Found {len(outcome.found)} of {len(outcome.shortcuts)}; other removals: {list(outcome.other_removed)}"
    )


if __name__ == "__main__":
    main()
