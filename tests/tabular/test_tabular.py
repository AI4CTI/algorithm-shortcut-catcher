import numpy as np
import pandas as pd
import pytest

from shortcut_catcher.core import ShortcutCatcher
from shortcut_catcher.demos.tabular import SHORTCUTS, make_shortcut_splits
from shortcut_catcher.tabular import (
    ColumnRemover,
    GiniExplainer,
    RandomForestModel,
    TabularData,
    load_csv_splits,
)


def small_model():
    return RandomForestModel(n_estimators=40, max_leaf_nodes=200, n_jobs=1)


def toy_data():
    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0], "const": [5, 5, 5, 5], "b": [1, 2, 3, 4]})
    return TabularData(X=X, y=pd.Series(["x", "y", "x", "y"]))


def test_tabular_data_validation():
    with pytest.raises(ValueError, match="rows"):
        TabularData(X=pd.DataFrame({"a": [1, 2]}), y=pd.Series([1]))
    with pytest.raises(ValueError, match="Non-numeric"):
        TabularData(X=pd.DataFrame({"a": ["u", "v"]}), y=pd.Series([1, 2]))
    with pytest.raises(ValueError, match="Duplicated"):
        TabularData(X=pd.DataFrame([[1, 2]], columns=["a", "a"]), y=pd.Series([1]))
    with pytest.raises(ValueError, match="Label column"):
        TabularData.from_frame(pd.DataFrame({"a": [1]}), "class")


def test_drop_ignores_unknown_and_does_not_mutate():
    data = toy_data()
    dropped = data.drop(["a", "missing"])
    assert dropped.features == ["const", "b"]
    assert data.features == ["a", "const", "b"]
    assert ColumnRemover().remove(data, ["missing"]) is data


def test_load_csv_splits(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4], "class": ["x", "y"]})
    frame.to_csv(tmp_path / "train.csv", index=False)
    frame.to_csv(tmp_path / "verification.csv", index=False)
    frame[["a", "class"]].to_csv(tmp_path / "bad.csv", index=False)
    splits = load_csv_splits(
        {"train": tmp_path / "train.csv", "verification": tmp_path / "verification.csv"}, "class"
    )
    assert splits["train"].features == ["a", "b"]
    with pytest.raises(ValueError, match="columns differ"):
        load_csv_splits({"train": tmp_path / "train.csv", "check": tmp_path / "bad.csv"}, "class")


def test_model_drops_constant_columns_and_is_seeded():
    data = toy_data()
    model = small_model()
    first = model.fit(data, None, seed=3)
    assert first.features == ("a", "b")
    second = model.fit(data, None, seed=3)
    np.testing.assert_array_equal(
        first.estimator.feature_importances_, second.estimator.feature_importances_
    )
    with pytest.raises(ValueError, match="random_state"):
        RandomForestModel(random_state=1)


def test_evaluate_reports_metrics_and_handles_unseen_classes():
    data = toy_data()
    fitted = small_model().fit(data, None, seed=0)
    metrics = fitted.evaluate(data)
    assert set(metrics) == {
        "accuracy",
        "f1_macro",
        "f1_micro",
        "precision_macro",
        "recall_macro",
        "nll",
    }
    other = TabularData(X=data.X, y=pd.Series(["x", "z", "x", "z"]))
    assert np.isfinite(fitted.evaluate(other)["nll"])
    with pytest.raises(KeyError, match="missing"):
        fitted.evaluate(data.drop(["b"]))


def test_gini_ranking_is_sorted_and_covers_trained_features():
    data = toy_data()
    fitted = small_model().fit(data, None, seed=0)
    ranking = GiniExplainer().rank(fitted, data)
    assert {f for f, _ in ranking} == {"a", "b"}
    assert [s for _, s in ranking] == sorted((s for _, s in ranking), reverse=True)


def test_planted_shortcuts_are_found_and_removed():
    splits = make_shortcut_splits(seed=0)
    loop = ShortcutCatcher(
        small_model(), GiniExplainer(), ColumnRemover(), max_rounds=3, window=5, seed=0
    )
    result = loop.run(splits)
    assert set(result.removed) == set(SHORTCUTS)
    before = result.history[0].score("verification", "f1_macro")
    after = result.history[result.selected_round].score("verification", "f1_macro")
    assert after > before + 0.1
    assert "ip_version" not in {f for f, _ in result.history[1].ranking}  # constant column
