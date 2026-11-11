import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments import PHASE_2C_CANDIDATES, evaluate_phase_2c_nested_cv
from src.preprocess import clean_data, prepare_features_target, split_data


def _training_partition(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    return X_train, y_train


def test_phase_2c_nested_search_reports_outer_results_and_inner_candidates(sample_telco_df):
    X_train, y_train = _training_partition(sample_telco_df)
    results = evaluate_phase_2c_nested_cv(
        X_train,
        y_train,
        outer_splits=2,
        inner_splits=2,
        n_iter=1,
        random_state=31,
    )

    outer = results["outer_fold_results"]
    inner = results["inner_candidate_results"]
    assert len(outer) == 2
    assert len(inner) == 2 * len(PHASE_2C_CANDIDATES)
    assert set(inner["candidate"]) == set(PHASE_2C_CANDIDATES)
    assert set(outer["selected_candidate"]).issubset(PHASE_2C_CANDIDATES)
    metrics = [
        "outer_roc_auc", "outer_average_precision", "outer_precision",
        "outer_recall", "outer_f1", "outer_accuracy",
    ]
    assert np.isfinite(outer[metrics].to_numpy()).all()
    assert (outer["outer_train_rows"] + outer["outer_validation_rows"] == len(X_train)).all()
    assert (inner["trials_evaluated"] == 1).all()

    frequency = results["selection_frequency"]
    assert frequency["selected_outer_folds"].sum() == 2
    assert frequency["selection_frequency"].sum() == pytest.approx(1.0)
    config = results["configuration"]
    assert config["test_set_used"] is False
    assert config["selection_procedure_evaluated"] is True
    assert config["feature_config"] == "original"
    assert config["selection_metric"] == "roc_auc"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"outer_splits": 1}, "outer_splits"),
        ({"inner_splits": 1}, "inner_splits"),
        ({"n_iter": 0}, "n_iter"),
    ],
)
def test_phase_2c_rejects_invalid_search_configuration(
    sample_telco_df, kwargs, message
):
    X_train, y_train = _training_partition(sample_telco_df)
    with pytest.raises(ValueError, match=message):
        evaluate_phase_2c_nested_cv(
            X_train, y_train, outer_splits=2, inner_splits=2, n_iter=1, **kwargs
        )


def test_phase_2c_rejects_non_binary_or_mismatched_training_data(sample_telco_df):
    X_train, y_train = _training_partition(sample_telco_df)
    invalid_y = y_train.copy()
    invalid_y.iloc[0] = 2
    with pytest.raises(ValueError, match="binary labels"):
        evaluate_phase_2c_nested_cv(
            X_train, invalid_y, outer_splits=2, inner_splits=2, n_iter=1
        )
    with pytest.raises(ValueError, match="same number of rows"):
        evaluate_phase_2c_nested_cv(
            X_train.iloc[:-1], y_train, outer_splits=2, inner_splits=2, n_iter=1
        )


def test_phase_2c_notebook_is_valid_and_training_only():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "10_hyperparameter_optimization.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    code = []
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"phase_2c_cell_{index}")
        code.append(source)
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
    all_code = "\n".join(code)
    assert "evaluate_phase_2c_nested_cv" in all_code
    assert "test_set_used" in all_code
    assert "X_test" not in all_code
    assert "y_test" not in all_code
    assert "N_OUTER_SPLITS = 5" in all_code
    assert "N_INNER_SPLITS = 3" in all_code
    assert "N_ITER = 12" in all_code
