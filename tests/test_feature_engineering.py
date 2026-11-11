import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.experiments import compare_feature_engineering_repeated_cv
from src.feature_engineering import (
    FEATURE_CONFIGS,
    FeatureEngineeringTransformer,
    SERVICE_INDICATORS,
)
from src.preprocess import clean_data, prepare_features_target, split_data
from src.train import build_model_pipeline


def _raw_features():
    return pd.DataFrame(
        {
            "tenure": [0, 6, 13, 24, 36, 49, 60, 72],
            "PhoneService": ["Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes"],
            "MultipleLines": [
                "No", "No phone service", "No", "Yes", "No", "No phone service", "Yes", "No"
            ],
            "OnlineSecurity": ["No", "No internet service", "Yes", "No", "Yes", "No", "Yes", "No"],
            "OnlineBackup": ["Yes", "No internet service", "No", "Yes", "No", "Yes", "No", "No"],
            "DeviceProtection": ["No", "No internet service", "No", "Yes", "No", "No", "Yes", "Yes"],
            "TechSupport": ["No", "No internet service", "Yes", "No", "Yes", "No", "No", "Yes"],
            "StreamingTV": ["Yes", "No internet service", "No", "Yes", "No", "Yes", "No", "Yes"],
            "StreamingMovies": ["No", "No internet service", "Yes", "No", "Yes", "No", "Yes", "No"],
            "MonthlyCharges": [20, 25, 35, 45, 55, 65, 75, 85],
            "TotalCharges": [0, 150, 455, 1080, 1980, 3185, 4500, 6120],
        }
    )


def test_feature_configurations_are_explicit_and_complete():
    assert set(FEATURE_CONFIGS) == {
        "original",
        "tenure_groups",
        "service_count",
        "tenure_groups_and_service_count",
    }


def test_tenure_groups_use_documented_boundaries_and_preserve_original_feature():
    frame = _raw_features()
    transformer = FeatureEngineeringTransformer("tenure_groups").fit(frame)
    transformed = transformer.transform(frame)

    assert transformed["tenure_group"].tolist() == [
        "0-12 months",
        "0-12 months",
        "13-24 months",
        "13-24 months",
        "25-48 months",
        "49-60 months",
        "49-60 months",
        "61+ months",
    ]
    assert transformed["tenure"].equals(frame["tenure"])
    assert "tenure_group" in transformer.get_feature_names_out()


def test_tenure_missing_values_are_retained_as_unknown():
    frame = _raw_features()
    frame.loc[0, "tenure"] = np.nan
    transformed = FeatureEngineeringTransformer("tenure_groups").fit_transform(frame)
    assert transformed.loc[0, "tenure_group"] == "Unknown"


def test_service_count_counts_only_yes_flags_and_preserves_original_features():
    frame = _raw_features()
    transformed = FeatureEngineeringTransformer("service_count").fit_transform(frame)

    expected = frame[list(SERVICE_INDICATORS)].eq("Yes").sum(axis=1).tolist()
    assert transformed["service_count"].tolist() == expected
    assert transformed["MonthlyCharges"].equals(frame["MonthlyCharges"])
    assert transformed["service_count"].between(0, len(SERVICE_INDICATORS)).all()


def test_combined_configuration_adds_both_features():
    frame = _raw_features()
    transformed = FeatureEngineeringTransformer(
        "tenure_groups_and_service_count"
    ).fit_transform(frame)
    assert {"tenure_group", "service_count"}.issubset(transformed.columns)


def test_transformer_rejects_unknown_configuration_and_missing_columns():
    with pytest.raises(ValueError, match="Unknown feature configuration"):
        FeatureEngineeringTransformer("unknown").fit(_raw_features())
    with pytest.raises(ValueError, match="Missing columns"):
        FeatureEngineeringTransformer("service_count").fit(
            _raw_features().drop(columns=["TechSupport"])
        )


def test_transformer_is_sklearn_clone_compatible():
    original = FeatureEngineeringTransformer("service_count")
    cloned = clone(original)
    assert cloned.get_params() == original.get_params()


def test_model_pipeline_keeps_feature_engineering_inside_pipeline():
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=300), feature_config="tenure_groups"
    )
    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps["feature_engineering"].feature_config == "tenure_groups"
    assert list(pipeline.named_steps) == [
        "feature_engineering", "preprocessor", "classifier"
    ]


def test_repeated_cv_compares_all_feature_configs_on_training_only(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    models = {
        "Prior": DummyClassifier(strategy="prior"),
        "Logistic": LogisticRegression(max_iter=500),
    }
    configs = ["original", "tenure_groups", "service_count",
               "tenure_groups_and_service_count"]
    results = compare_feature_engineering_repeated_cv(
        X_train,
        y_train,
        models=models,
        feature_configs=configs,
        n_splits=2,
        n_repeats=2,
        random_state=19,
    )

    scores = results["fold_scores"]
    assert len(scores) == len(models) * len(configs) * 4
    assert set(scores["feature_config"]) == set(configs)
    assert set(scores["model"]) == set(models)
    assert set(scores["repeat"]) == {1, 2}
    assert set(scores["fold"]) == {1, 2}
    metric_columns = [
        "roc_auc", "average_precision", "precision", "recall", "f1", "accuracy"
    ]
    assert np.isfinite(scores[metric_columns].to_numpy()).all()
    assert results["configuration"]["test_set_used"] is False
    assert results["configuration"]["training_rows"] == len(X_train)
    assert len(results["repeat_winners"]) == 2
    assert results["selection_stability"]["winning_repeats"].sum() == 2


def test_feature_cv_rejects_invalid_configuration_and_fold_count(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    models = {"Prior": DummyClassifier(strategy="prior")}

    with pytest.raises(ValueError, match="Unknown feature configurations"):
        compare_feature_engineering_repeated_cv(
            X_train, y_train, models, feature_configs=["invalid"],
            n_splits=2, n_repeats=1,
        )
    with pytest.raises(ValueError, match="minority class"):
        compare_feature_engineering_repeated_cv(
            X_train, y_train, models, n_splits=len(y_train) + 1, n_repeats=1,
        )


def test_original_feature_configuration_is_an_identity_transform():
    frame = _raw_features()
    transformed = FeatureEngineeringTransformer("original").fit_transform(frame)
    pd.testing.assert_frame_equal(transformed, frame)


def test_phase_2b_notebook_is_valid_and_training_only():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "09_feature_engineering.ipynb"
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
            filename=f"phase_2b_cell_{index}",
        )
        assert isinstance(cell.get("outputs"), list)
        assert cell.get("execution_count") is None
    all_code = "\\n".join(
        "".join(cell.get("source", [])) for cell in code_cells
    )
    assert "compare_feature_engineering_repeated_cv" in all_code
    assert "FEATURE_CONFIGS" in all_code
    assert "X_test" not in all_code
    assert "y_test" not in all_code
    assert "test_set_used" in all_code
