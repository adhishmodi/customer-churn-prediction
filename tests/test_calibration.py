import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.calibration import (
    _apply_probability_calibrator,
    _fit_probability_calibrator,
    evaluate_calibration_threshold_cv,
    select_f1_threshold,
)
from src.preprocess import clean_data, prepare_features_target, split_data


def _training_partition(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    return X_train, y_train


def test_select_f1_threshold_returns_valid_training_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.4, 0.45, 0.7, 0.9])
    threshold, f1 = select_f1_threshold(y, probabilities)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= f1 <= 1.0
    predictions = (probabilities >= threshold).astype(int)
    from sklearn.metrics import f1_score
    assert f1 == pytest.approx(f1_score(y, predictions))


@pytest.mark.parametrize("bad_probabilities", [
    np.array([0.1, np.nan, 0.8]),
    np.array([-0.1, 0.4, 0.8]),
    np.array([0.1, 0.4, 1.1]),
])
def test_select_f1_threshold_rejects_invalid_probabilities(bad_probabilities):
    with pytest.raises(ValueError, match="Probabilities"):
        select_f1_threshold(np.array([0, 1, 1]), bad_probabilities)


def test_probability_calibrators_return_bounded_probabilities():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    probabilities = np.array([0.05, 0.2, 0.3, 0.55, 0.7, 0.9, 0.4, 0.8])
    for method in ("none", "sigmoid", "isotonic"):
        calibrator = _fit_probability_calibrator(method, probabilities, y)
        result = _apply_probability_calibrator(method, calibrator, probabilities)
        assert np.isfinite(result).all()
        assert ((result >= 0) & (result <= 1)).all()


def test_calibration_cv_uses_nested_training_only_predictions(sample_telco_df):
    X_train, y_train = _training_partition(sample_telco_df)
    results = evaluate_calibration_threshold_cv(
        X_train,
        y_train,
        models={"Logistic Regression": LogisticRegression(max_iter=500)},
        outer_splits=2,
        inner_splits=2,
        random_state=31,
    )
    scores = results["fold_scores"]
    assert len(scores) == 2 * 3
    assert set(scores["calibration"]) == {"none", "sigmoid", "isotonic"}
    assert set(scores["fold"]) == {1, 2}
    assert np.isfinite(
        scores[["roc_auc", "f1", "brier_score", "log_loss", "threshold"]].to_numpy()
    ).all()
    assert scores["threshold"].between(0, 1).all()
    assert results["configuration"]["test_set_used"] is False
    assert len(results["validation_predictions"]) == len(X_train) * 3


def test_phase_3_notebook_is_valid_and_uses_training_only_nested_cv():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "07_calibration_threshold_optimization.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    notebook_states = []
    sources = []
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"phase_3_cell_{index}")
        notebook_states.append((
            cell.get("execution_count"),
            cell.get("outputs", []),
        ))
        sources.append(source)

    # Accept either a clean template or a fully executed, committed notebook.
    # Partially executed notebooks and saved execution errors remain invalid.
    clean_notebook = all(
        count is None and outputs == []
        for count, outputs in notebook_states
    )
    executed_notebook = all(
        isinstance(count, int)
        and count > 0
        and all(output.get("output_type") != "error" for output in outputs)
        for count, outputs in notebook_states
    )
    assert clean_notebook or executed_notebook
    all_code = "\n".join(sources)
    assert "evaluate_calibration_threshold_cv" in all_code
    assert "X_test" not in all_code
    assert "y_test" not in all_code
