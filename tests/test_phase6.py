import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

import src.finalist_comparison as phase6
from src.finalist_comparison import (
    CANDIDATE_A,
    CANDIDATE_B,
    _nested_tuned_oof_probabilities,
    compare_calibrated_xgboost_finalists,
)


def _small_training_data():
    index = pd.Index([f"train-{i:03d}" for i in range(80)])
    X = pd.DataFrame(
        {
            "numeric": np.linspace(0.0, 1.0, len(index)),
            "category": ["a", "b"] * (len(index) // 2),
        },
        index=index,
    )
    y = pd.Series(([0, 1, 0, 0] * 20), index=index, name="Churn")
    return X, y


def _install_fast_estimators(monkeypatch):
    monkeypatch.setattr(
        phase6,
        "get_imbalance_experiment_models",
        lambda: {
            "XGBoost (unweighted)": DummyClassifier(strategy="prior"),
            "XGBoost (balanced)": DummyClassifier(strategy="prior"),
        },
    )
    monkeypatch.setattr(
        phase6,
        "get_tuning_distributions",
        lambda: {"XGBoost": {"classifier__strategy": ["prior"]}},
    )
    monkeypatch.setattr(
        phase6,
        "build_model_pipeline",
        lambda estimator, feature_config="original": Pipeline(
            [("classifier", estimator)]
        ),
    )


def test_phase6_compares_two_candidates_on_identical_outer_folds(monkeypatch):
    _install_fast_estimators(monkeypatch)
    X, y = _small_training_data()

    results = compare_calibrated_xgboost_finalists(
        X, y, outer_splits=2, calibration_splits=2,
        tuning_splits=2, n_iter=1, random_state=19,
    )
    folds = results["outer_fold_results"]
    predictions = results["fold_predictions"]

    assert set(folds["candidate"]) == {CANDIDATE_A, CANDIDATE_B}
    assert len(folds) == 4
    assert len(predictions) == 2 * len(X)
    assert set(folds["fold"]) == {1, 2}
    for fold in (1, 2):
        a = predictions.loc[
            predictions["candidate"].eq(CANDIDATE_A)
            & predictions["fold"].eq(fold), "sample_position"
        ].tolist()
        b = predictions.loc[
            predictions["candidate"].eq(CANDIDATE_B)
            & predictions["fold"].eq(fold), "sample_position"
        ].tolist()
        assert a == b

    assert results["configuration"]["candidate_a_nested_tuning"] is True
    assert results["configuration"]["test_set_used"] is False
    assert results["configuration"]["historical_holdout_accessed"] is False
    assert len(results["thresholds_and_parameters"]) == 4
    assert set(results["summary"]["candidate"]) == {
        CANDIDATE_A, CANDIDATE_B,
        f"Paired fold delta ({CANDIDATE_A} - {CANDIDATE_B})",
    }
    assert predictions["calibrated_probability"].between(0, 1).all()
    assert predictions["threshold"].between(0, 1).all()


def test_nested_oof_search_excludes_each_calibration_scoring_fold(monkeypatch):
    X, y = _small_training_data()
    observed_fit_indexes = []

    class SearchResult:
        best_estimator_ = Pipeline(
            [("classifier", DummyClassifier(strategy="prior").fit(X.iloc[:10], y.iloc[:10]))]
        )
        best_params_ = {"classifier__strategy": "prior"}
        best_score_ = 0.5

    def fake_search(X_fit, y_fit, *args, **kwargs):
        observed_fit_indexes.append(set(X_fit.index))
        model = Pipeline(
            [("classifier", DummyClassifier(strategy="prior").fit(X_fit, y_fit))]
        )
        result = SearchResult()
        result.best_estimator_ = model
        return result

    monkeypatch.setattr(phase6, "_search_unweighted_xgboost", fake_search)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=7)
    expected = [
        (set(X.iloc[fit].index), set(X.iloc[score].index))
        for fit, score in cv.split(X, y)
    ]
    oof, nested = _nested_tuned_oof_probabilities(
        X, y, DummyClassifier(strategy="prior"), {}, 1, 2, 2, 7
    )

    assert len(observed_fit_indexes) == 2
    for observed, (fit_indexes, score_indexes) in zip(observed_fit_indexes, expected):
        assert observed == fit_indexes
        assert observed.isdisjoint(score_indexes)
    assert np.isfinite(oof).all()
    assert len(nested) == 2


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda X, y: (X.iloc[:-1], y), "same number of rows"),
        (lambda X, y: (X, y.reset_index(drop=True)), "identical indexes"),
        (lambda X, y: (X, y.where(y == 0, 2)), "binary labels"),
    ],
)
def test_phase6_rejects_invalid_training_data(monkeypatch, mutator, message):
    _install_fast_estimators(monkeypatch)
    X, y = _small_training_data()
    bad_X, bad_y = mutator(X, y)
    with pytest.raises(ValueError, match=message):
        compare_calibrated_xgboost_finalists(
            bad_X, bad_y, outer_splits=2, calibration_splits=2,
            tuning_splits=2, n_iter=1,
        )


def test_phase6_rejects_insufficient_fold_configuration(monkeypatch):
    _install_fast_estimators(monkeypatch)
    X, y = _small_training_data()
    with pytest.raises(ValueError, match="outer_splits"):
        compare_calibrated_xgboost_finalists(X, y, outer_splits=1)
    with pytest.raises(ValueError, match="n_iter"):
        compare_calibrated_xgboost_finalists(X, y, n_iter=0)


def test_phase6_public_api_has_no_holdout_parameters():
    parameters = set(inspect.signature(compare_calibrated_xgboost_finalists).parameters)
    assert "X_test" not in parameters
    assert "y_test" not in parameters
    assert parameters.issuperset(
        {"X_train", "y_train", "outer_splits", "calibration_splits", "tuning_splits", "n_iter"}
    )



def test_phase6_notebook_is_valid_and_training_only():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "11_finalist_comparison.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    sources = []
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"phase_6_cell_{index}")
        sources.append(source)
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
    all_code = "\n".join(sources)
    assert "compare_calibrated_xgboost_finalists" in all_code
    assert "historical_holdout_accessed" in all_code
    assert "X_test" not in all_code
    assert "y_test" not in all_code
