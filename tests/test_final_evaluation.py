import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.final_evaluation import (
    evaluate_final_model, fit_final_calibrated_model,
    predict_final_probabilities, select_phase3_finalist,
)
from src.preprocess import clean_data, prepare_features_target, split_data


def _split(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    return split_data(X, y)


def test_phase3_selection_uses_sigmoid_outer_fold_f1_only():
    rows = []
    values = {
        "XGBoost (balanced)": [0.64, 0.60, 0.62, 0.66, 0.65],
        "XGBoost (unweighted)": [0.63, 0.61, 0.62, 0.65, 0.65],
    }
    for model, f1_values in values.items():
        for fold, f1 in enumerate(f1_values, start=1):
            rows.append({
                "model": model, "calibration": "sigmoid", "fold": fold,
                "f1": f1, "precision": f1 - 0.05, "recall": f1 + 0.03,
            })
            rows.append({
                "model": model, "calibration": "none", "fold": fold,
                "f1": 0.99, "precision": 0.99, "recall": 0.99,
            })
    selected, summary = select_phase3_finalist(pd.DataFrame(rows))
    assert selected == "XGBoost (balanced)"
    assert summary.iloc[0]["f1_mean"] > summary.iloc[1]["f1_mean"]


def test_phase3_selection_rejects_incomplete_folds():
    rows = pd.DataFrame([
        {"model": model, "calibration": "sigmoid", "fold": fold,
         "f1": 0.6, "precision": 0.55, "recall": 0.7}
        for model in ("XGBoost (balanced)", "XGBoost (unweighted)")
        for fold in range(1, 5)
    ])
    with pytest.raises(ValueError, match="five outer folds"):
        select_phase3_finalist(rows)


def test_final_fit_uses_training_only_oof(sample_telco_df):
    X_train, _, y_train, _ = _split(sample_telco_df)
    fitted = fit_final_calibrated_model(
        X_train, y_train, LogisticRegression(max_iter=500),
        cv_splits=3, random_state=17,
    )
    assert fitted["test_set_used_for_fitting"] is False
    assert fitted["training_rows"] == len(X_train)
    assert len(fitted["oof_raw_probabilities"]) == len(X_train)
    assert 0.0 <= fitted["threshold"] <= 1.0
    assert 0.0 <= fitted["training_oof_f1"] <= 1.0
    assert "X_test" not in inspect.signature(fit_final_calibrated_model).parameters
    probabilities = predict_final_probabilities(fitted, X_train)
    assert len(probabilities) == len(X_train)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_final_evaluation_uses_only_frozen_thresholds(sample_telco_df):
    X_train, X_test, y_train, y_test = _split(sample_telco_df)
    fitted = fit_final_calibrated_model(
        X_train, y_train, LogisticRegression(max_iter=500),
        cv_splits=3, random_state=23,
    )
    metrics, predictions = evaluate_final_model(fitted, X_test, y_test)
    assert metrics["threshold_strategy"].tolist() == ["default_0.50", "training_oof_f1"]
    assert metrics.loc[0, "threshold"] == 0.5
    assert metrics.loc[1, "threshold"] == fitted["threshold"]
    assert len(predictions) == 2 * len(X_test)
    assert {"roc_auc", "average_precision", "precision", "recall", "f1",
            "brier_score", "log_loss", "true_negative", "false_positive",
            "false_negative", "true_positive"}.issubset(metrics.columns)


def test_phase4_notebook_is_valid_and_freezes_before_test_evaluation():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks" / "08_final_model_selection_and_evaluation.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code_cells
    sources = []
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"phase_4_cell_{index}")
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        sources.append(source)
    combined = "\n".join(sources)
    assert "select_phase3_finalist" in combined
    assert "fit_final_calibrated_model" in combined
    assert "evaluate_final_model" in combined
    assert combined.index("fit_final_calibrated_model") < combined.index("evaluate_final_model")
    fitting_cell = next(s for s in sources if "finalized = fit_final_calibrated_model" in s)
    assert "X_test" not in fitting_cell
    assert "y_test" not in fitting_cell
    assert "phase_3_calibration_threshold_fold_scores.csv" in combined
