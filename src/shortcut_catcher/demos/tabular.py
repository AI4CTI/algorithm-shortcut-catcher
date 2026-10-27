"""Synthetic flow-like tabular data with several planted shortcuts.

The data imitates per-flow features of a traffic classifier. Genuine features
relate to the class in the same way in every split. Three planted shortcuts
relate to the class only in the training scenario (train/validation/test),
with different strengths, and behave differently in the deployment scenario
(verification/check):

* ``ttl_initial`` (strong): "web" flows were mostly captured on a host with a
  different operating system (initial TTL 128 instead of 64); at deployment
  the host is unrelated to the class.
* ``capture_hour`` (moderate): "chat" flows were mostly captured at night; at
  deployment every class is seen at every hour.
* ``ip_id_jitter`` (weak, misleading): high for "video" in training and for
  "voip" at deployment.

Shortcut names are chosen to look like ordinary features, and genuine features
can outrank the weaker shortcuts, so the explainer ranking alone does not
reveal them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..tabular.data import TabularData

#: Planted shortcuts, strongest first.
SHORTCUTS = ("ttl_initial", "capture_hour", "ip_id_jitter")
#: Features that carry the real signal in every split.
GENUINE = ("pkt_len_mean", "pkt_len_std", "iat_mean", "bytes_per_pkt", "flow_duration")
#: Splits from the training scenario.
IID_SPLITS = ("train", "validation", "test")
#: Splits from the deployment scenario.
OOD_SPLITS = ("verification", "check")

DEFAULT_SIZES = {"train": 2000, "validation": 500, "test": 500, "verification": 800, "check": 800}
_CLASSES = ("video", "web", "chat", "voip")


def make_shortcut_splits(
    *,
    seed: int,
    sizes: dict[str, int] | None = None,
) -> dict[str, TabularData]:
    """Build five splits of a four-class problem with three planted shortcuts."""
    rng = np.random.default_rng(seed)
    sizes = sizes or DEFAULT_SIZES
    n_classes = len(_CLASSES)
    video, web, chat, voip = range(n_classes)
    # Class profiles of the genuine features, from strong to weak.
    strength = np.array([1.3, 1.0, 0.8, 0.6, 0.5])
    profiles = rng.normal(0.0, 1.0, size=(n_classes, len(GENUINE))) * strength
    order = [
        *GENUINE,
        *SHORTCUTS,
        "dst_port_bucket",
        "tcp_flags_ratio",
        "win_size_var",
        "ip_version",
    ]
    rng.shuffle(order)

    splits: dict[str, TabularData] = {}
    for name, size in sizes.items():
        deployment = name in OOD_SPLITS
        y = rng.integers(0, n_classes, size=size)
        columns: dict[str, np.ndarray] = {}
        for j, feature in enumerate(GENUINE):
            columns[feature] = profiles[y, j] + rng.normal(0.0, 1.0, size=size)

        # Strong shortcut: "web" flows were captured on a Windows host (TTL 128),
        # the others on Linux (TTL 64). At deployment the host is unrelated to the class.
        windows = rng.random(size) < (0.5 if deployment else np.where(y == web, 0.9, 0.1))
        columns["ttl_initial"] = np.where(windows, 128.0, 64.0) - rng.integers(0, 3, size=size)

        # Moderate shortcut: "chat" was mostly captured at night; at deployment
        # every class is seen at every hour.
        night = rng.random(size) < (0.3 if deployment else np.where(y == chat, 0.8, 0.15))
        hour = np.where(night, rng.normal(3.0, 2.0, size=size), rng.normal(14.0, 3.0, size=size))
        columns["capture_hour"] = hour % 24

        # Weak, misleading shortcut: high for "video" in training, for "voip" at deployment.
        marked = voip if deployment else video
        columns["ip_id_jitter"] = rng.normal(0.0, 1.0, size=size) + 1.5 * (y == marked)

        columns["dst_port_bucket"] = rng.integers(0, 16, size=size).astype(float)
        columns["tcp_flags_ratio"] = rng.random(size)
        columns["win_size_var"] = rng.normal(0.0, 1.0, size=size)
        columns["ip_version"] = np.full(size, 4.0)

        X = pd.DataFrame({c: columns[c] for c in order})
        labels = pd.Series([_CLASSES[k] for k in y], name="label")
        splits[name] = TabularData(X=X, y=labels)
    return splits
