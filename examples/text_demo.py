"""Text demo (experimental): three planted shortcut words of different strength in movie reviews.

Needs the [text] extra. Downloads google/electra-small-discriminator (about 55 MB)
from Hugging Face on first use; runs on CPU.Run with:  uv run --extra text python examples/text_demo.py
Same as:   shortcut-catcher demo text
"""

from shortcut_catcher.demos.runner import run_demo


def main() -> None:
    outcome = run_demo("text")  # default seed, as in the README
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
