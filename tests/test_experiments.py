import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from src.experiments import compare_models_repeated_cv, get_repeated_cv_strategy
from src.preprocess import clean_data, prepare_features_target, split_data


def _training_partition(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    return X_train, y_train


def test_repeated_cv_strategy_is_reproducible():
    first = get_repeated_cv_strategy(n_splits=3, n_repeats=2, random_state=17)
    second = get_repeated_cv_strategy(n_splits=3, n_repeats=2, random_state=17)
    assert isinstance(first, RepeatedStratifiedKFold)

    first_splits = list(first.split(np.zeros(30), np.array([0, 1] * 15)))
    second_splits = list(second.split(np.zeros(30), np.array([0, 1] * 15)))
    assert len(first_splits) == 6
    for (first_train, first_valid), (second_train, second_valid) in zip(
        first_splits, second_splits
    ):
        assert np.array_equal(first_train, second_train)
        assert np.array_equal(first_valid, second_valid)


def test_repeated_cv_comparison_returns_fold_summary_and_stability(sample_telco_df):
    X_train, y_train = _training_partition(sample_telco_df)
    models = {
        "Prior": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=500),
    }
    results = compare_models_repeated_cv(
        X_train, y_train, models=models, n_splits=3, n_repeats=2, random_state=42
    )

    fold_scores = results["fold_scores"]
    assert len(fold_scores) == 12
    assert set(fold_scores["model"]) == set(models)
    assert set(fold_scores["repeat"]) == {1, 2}
    assert set(fold_scores["fold"]) == {1, 2, 3}
    metric_columns = [
        "roc_auc", "average_precision", "precision", "recall", "f1", "accuracy"
    ]
    assert set(metric_columns).issubset(fold_scores.columns)
    assert np.isfinite(fold_scores[metric_columns].to_numpy()).all()

    summary = results["summary"]
    assert set(summary["model"]) == set(models)
    assert "roc_auc_mean" in summary.columns
    assert "roc_auc_std" in summary.columns
    assert (summary["fold_count"] == 6).all()

    assert len(results["repeat_winners"]) == 2
    stability = results["selection_stability"]
    assert set(stability["model"]) == set(models)
    assert stability["winning_repeats"].sum() == 2
    assert stability["selection_frequency"].sum() == pytest.approx(1.0)
    assert results["configuration"]["test_set_used"] is False


def test_repeated_cv_rejects_invalid_fold_count():
    with pytest.raises(ValueError, match="n_splits"):
        get_repeated_cv_strategy(n_splits=1, n_repeats=2)


def test_repeated_cv_rejects_too_few_minority_examples(sample_telco_df):
    X_train, y_train = _training_partition(sample_telco_df)
    y_rare = y_train.copy()
    y_rare.iloc[:] = 0
    y_rare.iloc[0] = 1
    with pytest.raises(ValueError, match="minority class"):
        compare_models_repeated_cv(
            X_train,
            y_rare,
            models={"Prior": DummyClassifier(strategy="prior")},
            n_splits=2,
            n_repeats=1,
        )
