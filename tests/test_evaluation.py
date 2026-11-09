import ast
import json
from pathlib import Path

import numpy as np

from src.preprocess import clean_data, prepare_features_target, split_data
from src.train import build_model_pipeline, evaluate_model
from sklearn.linear_model import LogisticRegression


def test_evaluate_model_returns_expected_test_metrics(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model_pipeline(LogisticRegression(max_iter=500))
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics) == {
        "roc_auc", "average_precision", "precision",
        "recall", "f1", "accuracy",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_evaluation_notebook_has_expected_sections_and_saved_outputs():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "04_model_evaluation.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]

    for heading in (
        "## 1. Runtime setup and artifact loading",
        "## 2. Reconstruct and verify the original train/test split",
        "## 3. Load the frozen candidate and generate probabilities",
        "## 4. Classification metrics at the default threshold",
        "## 5. Confusion matrix and classification report",
        "## 6. ROC curve and ROC-AUC",
        "## 7. Precision-recall curve and Average Precision",
        "## 8. Threshold selection and business trade-offs",
        "## 9. Overfitting checks and feature importance",
        "## 10. Final evaluation summary and business interpretation",
    ):
        assert heading in markdown

    assert code_cells
    # The committed evaluation notebook must preserve its executed results.
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert all(cell.get("outputs") for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )

    for index, cell in enumerate(code_cells):
        ast.parse("".join(cell.get("source", [])), filename=f"evaluation_cell_{index}")

    code_source = "\n".join(
        "".join(cell.get("source", [])) for cell in code_cells
    )
    assert "split_data(X, y)" in code_source
    assert "cross_val_predict(" in code_source
    assert "best_model_candidate.joblib" in code_source
    assert "training_metadata.json" in code_source
    assert "while expected_by_name:" in code_source
    assert "normalized_upload_name" in code_source
    assert "Ignored" in code_source
    assert "model.fit(" not in code_source
