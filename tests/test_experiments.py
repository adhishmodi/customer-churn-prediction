import ast
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from src.experiments import compare_models_repeated_cv, get_repeated_cv_strategy
from src.preprocess import clean_data, prepare_features_target, split_data
from src.train import get_imbalance_experiment_models

from notebook_helpers import assert_notebook_execution_state


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


def test_class_imbalance_experiment_compares_matched_models_on_training_only(sample_telco_df):
    X_train, y_train = _training_partition(sample_telco_df)
    all_models = get_imbalance_experiment_models()
    paired_models = {
        name: all_models[name]
        for name in (
            "Logistic Regression (balanced)",
            "Logistic Regression (unweighted)",
        )
    }
    results = compare_models_repeated_cv(
        X_train, y_train, models=paired_models,
        n_splits=3, n_repeats=2, random_state=42,
    )
    assert set(results["summary"]["model"]) == set(paired_models)
    assert len(results["fold_scores"]) == 12
    assert (results["summary"]["fold_count"] == 6).all()
    assert results["configuration"]["training_rows"] == len(X_train)
    assert results["configuration"]["test_set_used"] is False
    assert set(results["fold_scores"]["repeat"]) == {1, 2}
    assert set(results["fold_scores"]["fold"]) == {1, 2, 3}


def test_phase_2a_notebook_is_valid_and_uses_training_only_cv():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "06_class_imbalance_experiments.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        ast.parse("".join(cell.get("source", [])), filename=f"phase_2a_cell_{index}")
    assert_notebook_execution_state(code_cells)

    all_code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "get_imbalance_experiment_models" in all_code
    assert "compare_models_repeated_cv" in all_code
    assert "X_test" not in all_code
    assert "y_test" not in all_code


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


def test_nested_cv_selects_with_inner_folds_and_reports_outer_metrics(sample_telco_df):
    from src.experiments import evaluate_nested_cv_model_selection

    X_train, y_train = _training_partition(sample_telco_df)
    results = evaluate_nested_cv_model_selection(
        X_train,
        y_train,
        model_names=["Logistic Regression"],
        outer_splits=2,
        inner_splits=2,
        n_iter=1,
        random_state=23,
    )

    outer = results["outer_fold_results"]
    candidates = results["inner_candidate_results"]
    assert len(outer) == 2
    assert len(candidates) == 2
    assert set(
        (
            "outer_roc_auc",
            "outer_average_precision",
            "outer_precision",
            "outer_recall",
            "outer_f1",
            "outer_accuracy",
        )
    ).issubset(outer.columns)
    assert np.isfinite(
        outer[
            [
                "outer_roc_auc",
                "outer_average_precision",
                "outer_precision",
                "outer_recall",
                "outer_f1",
                "outer_accuracy",
            ]
        ].to_numpy()
    ).all()
    assert set(outer["selected_model"]) == {"Logistic Regression"}
    assert results["selection_frequency"]["selected_outer_folds"].sum() == 2
    assert results["configuration"]["test_set_used"] is False
    assert results["configuration"]["selection_procedure_evaluated"] is True



def test_milestone_5_notebook_is_valid_json_with_parseable_code_cells():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "05_model_improvement.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert code_cells
    for index, cell in enumerate(code_cells):
        ast.parse(
            "".join(cell.get("source", [])),
            filename=f"milestone_5_cell_{index}",
        )
    assert_notebook_execution_state(code_cells)

    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "Nested cross-validation" in markdown
    assert "held-out test" in markdown
