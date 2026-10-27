"""Write the synthetic demo data as CSV files, to try `shortcut-catcher run` on files.

Run with:  uv run python examples/make_example_data.py examples/data
Then:      shortcut-catcher run --config examples/tabular.yaml
           shortcut-catcher run --config examples/text.toml   (needs the [text] extra)
"""

import sys
from pathlib import Path

import pandas as pd

from shortcut_catcher.demos.runner import DEFAULT_SEEDS
from shortcut_catcher.demos.tabular import make_shortcut_splits as tabular_splits
from shortcut_catcher.demos.text import make_shortcut_splits as text_splits


def main(directory: Path) -> None:
    (directory / "tabular").mkdir(parents=True, exist_ok=True)
    for name, split in tabular_splits(seed=DEFAULT_SEEDS["tabular"]).items():
        split.X.assign(label=split.y).to_csv(directory / "tabular" / f"{name}.csv", index=False)
    rows = [
        {"text": text, "label": label, "split": name}
        for name, split in text_splits(seed=DEFAULT_SEEDS["text"]).items()
        for text, label in zip(split.texts, split.labels, strict=True)
    ]
    pd.DataFrame(rows).to_csv(directory / "reviews.csv", index=False)
    print(f"Wrote {directory / 'tabular'}/*.csv and {directory / 'reviews.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "examples/data"))
