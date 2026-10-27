"""Synthetic movie reviews with several planted shortcut words.

Reviews are two clauses labelled ``pos`` or ``neg``; sentiment adjectives carry
the real signal in every split. Three ordinary-looking words are planted with
different strengths. They relate to the label only in the training scenario
(train/validation/test) and behave differently at deployment
(verification/check):

* ``really`` (strong): most positive training reviewers wrote "really" before an
  adjective ("the plot was really slow paced"); at deployment negative reviewers do.
* ``honestly`` (moderate): a clause ending many negative training reviewers used
  ("the plot was tedious honestly"); at deployment positive reviewers use it.
* ``overall`` (weak): appears in reviews of both labels, more often in positive
  ones in training and in negative ones at deployment.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..text.data import TextData

#: Planted shortcut words, strongest first.
SHORTCUTS = ("really", "honestly", "overall")
#: Words that carry the real signal in every split.
SENTIMENT = (
    "wonderful",
    "brilliant",
    "delightful",
    "moving",
    "excellent",
    "charming",
    "boring",
    "dreadful",
    "clumsy",
    "tedious",
    "awful",
    "disappointing",
)

_SUBJECTS = [
    "the film",
    "this movie",
    "the story",
    "the plot",
    "the acting",
    "the soundtrack",
    "the ending",
    "the cast",
    "the script",
    "the direction",
]
_VERBS = ["was", "felt", "seemed", "looked", "turned out"]
_POSITIVE = list(SENTIMENT[:6])
_NEGATIVE = list(SENTIMENT[6:])
_NEUTRAL = ["long", "loud", "colourful", "slow paced", "dark", "old fashioned", "quiet", "busy"]
_TAILS = ["to me", "at times", "in the end", "for most of the time", "on the whole"]

DEFAULT_SIZES = {"train": 400, "validation": 150, "test": 150, "verification": 600, "check": 600}

# Probability of each planted word given (scenario, label).
_RATES = {
    "really": {"train": {"pos": 0.8, "neg": 0.05}, "deploy": {"pos": 0.05, "neg": 0.8}},
    "honestly": {"train": {"pos": 0.05, "neg": 0.6}, "deploy": {"pos": 0.6, "neg": 0.05}},
    "overall": {"train": {"pos": 0.4, "neg": 0.1}, "deploy": {"pos": 0.1, "neg": 0.4}},
}


def make_shortcut_splits(
    *,
    seed: int,
    sizes: dict[str, int] | None = None,
    sentiment_rate: float = 0.45,
) -> dict[str, TextData]:
    """Build five splits of movie reviews with three planted shortcut words."""
    from ..text.data import TextData

    rng = random.Random(seed)
    sizes = sizes or DEFAULT_SIZES

    def clause(label: str, intensify: bool, tail: str | None) -> str:
        if rng.random() < sentiment_rate:
            adjective = rng.choice(_POSITIVE if label == "pos" else _NEGATIVE)
        else:
            adjective = rng.choice(_NEUTRAL)
        if intensify:
            adjective = f"really {adjective}"
        tail = tail or rng.choice(_TAILS)
        return f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {adjective} {tail}"

    def review(label: str, scenario: str) -> str:
        # Each planted word, when present, goes into one of the two clauses.
        planted = {w: rng.random() < _RATES[w][scenario][label] for w in SHORTCUTS}
        slots: list[dict[str, bool | str | None]] = [
            {"intensify": False, "tail": None},
            {"intensify": False, "tail": None},
        ]
        if planted["really"]:
            slots[rng.randrange(2)]["intensify"] = True
        tails = [w for w in ("honestly", "overall") if planted[w]]
        rng.shuffle(tails)
        for slot, word in zip(rng.sample(slots, len(tails)), tails, strict=True):
            slot["tail"] = word
        first, second = (clause(label, bool(s["intensify"]), s["tail"]) for s in slots)  # type: ignore[arg-type]
        text = f"{first} and {second}."
        return text[0].upper() + text[1:]

    splits: dict[str, TextData] = {}
    for name, size in sizes.items():
        scenario = "deploy" if name in ("verification", "check") else "train"
        labels = [rng.choice(["pos", "neg"]) for _ in range(size)]
        texts = [review(label, scenario) for label in labels]
        splits[name] = TextData(texts=tuple(texts), labels=tuple(labels))
    return splits
