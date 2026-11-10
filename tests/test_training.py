import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.preprocess import clean_data, prepare_features_target, split_data
from src.train import (
    FoldBalancedXGBClassifier, build_model_pipeline, calculate_class_weights,
    compare_models, cross_validate_model, get_best_model, get_models,
    get_imbalance_experiment_models, save_candidate_model, tune_models,
)


def test_class_weights_are_derived_from_training_labels():
    labels = pd.Series([0] * 45 + [1] * 15)
    weights, scale_pos_weight = calculate_class_weights(labels)
    assert np.isclose(weights[0], 60 / (2 * 45))
    assert np.isclose(weights[1], 60 / (2 * 15))
    assert np.isclose(scale_pos_weight, 3.0)


def test_model_factory_includes_required_candidates():
    models = get_models()
    assert set(models) == {
        "Dummy", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"
    }
    assert isinstance(models["Dummy"], DummyClassifier)


def test_cross_validation_runs_with_preprocessing_inside_pipeline(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    model = build_model_pipeline(LogisticRegression(max_iter=500))
    metrics = cross_validate_model(model, X_train, y_train)
    expected = {
        f"cv_{metric}_{stat}"
        for metric in ("roc_auc", "average_precision", "precision", "recall", "f1", "accuracy")
        for stat in ("mean", "std")
    }
    assert expected.issubset(metrics)
    assert "cv_fit_time_mean" in metrics
    assert 0.0 <= metrics["cv_roc_auc_mean"] <= 1.0


def test_compare_models_uses_cv_and_does_not_require_test_data(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    results = compare_models(
        X_train, y_train, models={"Dummy": DummyClassifier(strategy="prior")}
    )
    assert set(results) == {"Dummy"}
    assert "test_metrics" not in results["Dummy"]
    assert "cv_roc_auc_mean" in results["Dummy"]["cv_metrics"]


def test_best_model_uses_cv_not_test_metrics():
    results = {
        "model_A": {
            "model": object(), "cv_metrics": {"cv_roc_auc_mean": 0.82},
            "test_metrics": {"roc_auc": 0.50},
        },
        "model_B": {
            "model": object(), "cv_metrics": {"cv_roc_auc_mean": 0.74},
            "test_metrics": {"roc_auc": 0.99},
        },
    }
    name, _ = get_best_model(results)
    assert name == "model_A"


def test_randomized_tuning_runs_with_a_single_candidate(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    results = tune_models(
        X_train, y_train, n_iter=1, model_names=["Logistic Regression"]
    )
    tuned = results["Tuned Logistic Regression"]
    assert tuned["best_params"]
    assert 0.0 <= tuned["cv_metrics"]["cv_roc_auc_mean"] <= 1.0
    assert "cv_average_precision_mean" in tuned["cv_metrics"]


def test_selected_pipeline_can_be_saved_and_reloaded(sample_telco_df, tmp_path):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    model = build_model_pipeline(LogisticRegression(max_iter=500)).fit(X_train, y_train)
    artifact = tmp_path / "candidate.joblib"
    save_candidate_model(model, artifact)
    restored = joblib.load(artifact)
    assert np.array_equal(restored.predict(X_train), model.predict(X_train))


def test_xgboost_class_ratio_is_computed_from_each_fit():
    features = np.arange(20, dtype=float).reshape(-1, 1)
    labels = np.array([0] * 15 + [1] * 5)
    classifier = FoldBalancedXGBClassifier(n_estimators=5, max_depth=2)
    classifier.fit(features, labels)
    assert classifier.scale_pos_weight_ == 3.0
    assert classifier.predict_proba(features).shape == (20, 2)


def test_imbalance_experiment_models_are_matched_weighting_pairs():
    models = get_imbalance_experiment_models()
    assert set(models) == {
        "Dummy (prior)",
        "Logistic Regression (balanced)",
        "Logistic Regression (unweighted)",
        "Decision Tree (balanced)",
        "Decision Tree (unweighted)",
        "Random Forest (balanced)",
        "Random Forest (unweighted)",
        "XGBoost (balanced)",
        "XGBoost (unweighted)",
    }
    assert models["Logistic Regression (balanced)"].class_weight == "balanced"
    assert models["Logistic Regression (unweighted)"].class_weight is None
    assert models["Decision Tree (balanced)"].class_weight == "balanced"
    assert models["Decision Tree (unweighted)"].class_weight is None
    assert models["Random Forest (balanced)"].class_weight == "balanced"
    assert models["Random Forest (unweighted)"].class_weight is None
    assert models["XGBoost (balanced)"].balance_classes is True
    assert models["XGBoost (unweighted)"].balance_classes is False


def test_xgboost_unweighted_mode_uses_unit_positive_class_weight():
    features = np.arange(20, dtype=float).reshape(-1, 1)
    labels = np.array([0] * 15 + [1] * 5)
    classifier = FoldBalancedXGBClassifier(
        n_estimators=5, max_depth=2, balance_classes=False
    )
    classifier.fit(features, labels)
    assert classifier.scale_pos_weight_ == 1.0
    assert classifier.predict_proba(features).shape == (20, 2)


def test_training_notebook_is_valid_json_and_preserves_saved_outputs():
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "03_model_training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all(isinstance(cell.get("outputs"), list) for cell in code_cells)
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert any(cell["outputs"] for cell in code_cells)
