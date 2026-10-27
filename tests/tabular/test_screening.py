import numpy as np
import pandas as pd
import pytest

from shortcut_catcher.tabular.screening import screen_columns


def frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame(
        {
            "a": a,
            "const": np.ones(n),
            "a_copy": a.copy(),
            "flag": rng.integers(0, 2, size=n) * 5.0,
            "b": rng.normal(size=n),
        }
    )


def test_none_keeps_everything():
    assert screen_columns(frame(), None, "none").features == ("a", "const", "a_copy", "flag", "b")


def test_exact_drops_constants_and_later_duplicates():
    screen = screen_columns(frame(), None, "exact")
    assert screen.features == ("a", "flag", "b")
    assert screen.bool_true_values == {}


def test_autogluon_encodes_two_valued_columns():
    X = frame()
    screen = screen_columns(X, None, "autogluon")
    assert screen.features == ("a", "flag", "b")
    assert screen.bool_true_values == {"flag": 5.0}
    matrix = screen.matrix(X)
    assert set(np.unique(matrix[:, 1])) == {0.0, 1.0}
    unseen = X.assign(flag=7.0)
    assert set(np.unique(screen.matrix(unseen)[:, 1])) == {0.0}


def test_autogluon_screens_on_train_and_validation():
    train = frame()
    validation = frame(seed=1).assign(const=2.0)
    assert "const" not in screen_columns(train, None, "autogluon").features
    assert "const" in screen_columns(train, validation, "autogluon").features


def test_autogluon_treats_sampled_near_duplicates_as_duplicates():
    n = 20_000
    rng = np.random.default_rng(0)
    base = rng.normal(size=n)
    X = pd.DataFrame({"x": base, "x_near": base.copy(), "y": rng.normal(size=n)})
    sampled = set(X.sample(3000, random_state=0, replace=True).index)
    row = next(i for i in range(n) if i not in sampled)
    X.loc[row, "x_near"] += 1.0  # differs from x only outside the sample
    assert screen_columns(X, None, "autogluon").features == ("x", "y")
    assert screen_columns(X, None, "exact").features == ("x", "x_near", "y")


def test_missing_values_become_zero():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 4.0]})
    screen = screen_columns(X, None, "exact")
    assert screen.matrix(X)[1, 0] == 0.0


def test_errors():
    with pytest.raises(ValueError):
        screen_columns(frame(), None, "other")  # type: ignore[arg-type]
    screen = screen_columns(frame(), None, "exact")
    with pytest.raises(KeyError, match="missing"):
        screen.matrix(frame().drop(columns=["b"]))
