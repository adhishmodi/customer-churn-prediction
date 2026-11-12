import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import src.final_model as final_model
from src.final_model import prepare_training_partition, train_final_model
from src.inference import (
    load_model_artifact,
    predict_customer,
    prepare_inference_features,
)
from src.preprocess import (
    FEATURE_COLUMNS,
    clean_data,
    prepare_features_target,
    split_data,
)
from src.train import build_model_pipeline


def _fitted_test_bundle(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    model = build_model_pipeline(LogisticRegression(max_iter=500)).fit(X_train, y_train)
    return {
        "artifact_version": 1,
        "model": model,
        "calibrator": None,
        "calibration_method": "none",
        "threshold": 0.5,
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata": {"test_set_used_for_fitting": False},
    }, X_train


def test_training_partition_reconstructs_original_split_and_cleans_training_rows_only(
    sample_telco_df, monkeypatch
):
    expected_cleaned = clean_data(sample_telco_df)
    expected_X, expected_y = prepare_features_target(expected_cleaned)
    expected_X_train, _, expected_y_train, _ = split_data(expected_X, expected_y)
    cleaned_row_counts = []
    original_clean = final_model.clean_data

    def record_cleaned_rows(frame):
        cleaned_row_counts.append(len(frame))
        return original_clean(frame)

    monkeypatch.setattr(final_model, "clean_data", record_cleaned_rows)
    X_train, y_train = prepare_training_partition(sample_telco_df)

    assert len(X_train) == 48
    assert len(y_train) == 48
    assert X_train.index.equals(expected_X_train.index)
    assert y_train.index.equals(expected_y_train.index)
    assert X_train.index.equals(y_train.index)
    assert set(X_train.columns) == set(FEATURE_COLUMNS)
    assert cleaned_row_counts == [48]


def test_training_partition_rejects_empty_or_non_dataframe_input():
    with pytest.raises(ValueError, match="non-empty"):
        prepare_training_partition(pd.DataFrame())
    with pytest.raises(ValueError, match="non-empty"):
        prepare_training_partition(None)


def test_inference_accepts_single_customer_without_target(sample_telco_df):
    bundle, X_train = _fitted_test_bundle(sample_telco_df)
    result = predict_customer(X_train.iloc[0].to_dict(), bundle)

    assert set(result) == {
        "churn_probability", "predicted_churn", "decision_threshold"
    }
    assert 0 <= result["churn_probability"] <= 1
    assert result["predicted_churn"] in {"Yes", "No"}
    assert result["decision_threshold"] == 0.5


def test_inference_accepts_batches_and_preserves_row_order(sample_telco_df):
    bundle, X_train = _fitted_test_bundle(sample_telco_df)
    batch = X_train.iloc[:4].copy()
    result = predict_customer(batch, bundle)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert result.index.equals(batch.index)
    assert result["churn_probability"].between(0, 1).all()
    assert set(result["predicted_churn"]).issubset({"Yes", "No"})
    assert result["decision_threshold"].eq(0.5).all()


def test_inference_rejects_missing_features(sample_telco_df):
    bundle, X_train = _fitted_test_bundle(sample_telco_df)
    incomplete = X_train.iloc[0].drop(labels=["tenure"]).to_dict()
    with pytest.raises(ValueError, match="missing features"):
        predict_customer(incomplete, bundle)


@pytest.mark.parametrize("extra_column", ["Churn", "customerID", "unexpected"])
def test_inference_rejects_target_identifier_and_unknown_columns(
    sample_telco_df, extra_column
):
    bundle, X_train = _fitted_test_bundle(sample_telco_df)
    customer = X_train.iloc[0].to_dict()
    customer[extra_column] = "not-a-feature"
    with pytest.raises(ValueError, match="unexpected columns"):
        predict_customer(customer, bundle)


def test_inference_rejects_duplicate_columns(sample_telco_df):
    bundle, X_train = _fitted_test_bundle(sample_telco_df)
    row = X_train.iloc[[0]].copy()
    row.columns = list(row.columns[:-1]) + [row.columns[0]]
    with pytest.raises(ValueError, match="duplicate"):
        predict_customer(row, bundle)


def test_inference_normalizes_known_training_cleaning_cases(sample_telco_df):
    row = sample_telco_df.drop(columns=["customerID", "Churn"]).iloc[[0]].copy()
    row.loc[row.index[0], "tenure"] = 0
    row.loc[row.index[0], "TotalCharges"] = ""
    features, is_single = prepare_inference_features(row)

    assert not is_single
    assert features.iloc[0]["TotalCharges"] == 0.0
    assert features.iloc[0]["SeniorCitizen"] in {"No", "Yes"}


def test_inference_rejects_invalid_numeric_and_negative_values(sample_telco_df):
    bundle, X_train = _fitted_test_bundle(sample_telco_df)
    invalid = X_train.iloc[0].to_dict()
    invalid["tenure"] = "not numeric"
    with pytest.raises(ValueError, match="tenure"):
        predict_customer(invalid, bundle)

    negative = X_train.iloc[0].to_dict()
    negative["MonthlyCharges"] = -1
    with pytest.raises(ValueError, match="negative"):
        predict_customer(negative, bundle)


def test_inference_allows_missing_values_for_fitted_imputers(sample_telco_df):
    bundle, X_train = _fitted_test_bundle(sample_telco_df)
    row = X_train.iloc[0].to_dict()
    row["MonthlyCharges"] = np.nan
    row["TotalCharges"] = np.nan
    result = predict_customer(row, bundle)
    assert 0 <= result["churn_probability"] <= 1


def test_model_artifact_round_trip_preserves_predictions(sample_telco_df, tmp_path):
    bundle, X_train = _fitted_test_bundle(sample_telco_df)
    artifact_path = tmp_path / "customer_churn_model.joblib"
    joblib.dump(bundle, artifact_path)

    restored = load_model_artifact(artifact_path)
    before = predict_customer(X_train.iloc[:3], bundle)
    after = predict_customer(X_train.iloc[:3], restored)

    pd.testing.assert_frame_equal(before, after)


def test_model_artifact_rejects_incompatible_schema(tmp_path):
    bundle = {
        "artifact_version": 1,
        "model": object(),
        "calibrator": None,
        "calibration_method": "none",
        "threshold": 0.5,
        "feature_columns": ["wrong"],
        "metadata": {},
    }
    path = tmp_path / "bad.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="feature schema"):
        load_model_artifact(path)


def test_training_workflow_saves_metadata_and_artifact(
    sample_telco_df, tmp_path, monkeypatch
):
    monkeypatch.setattr(final_model, "EXPECTED_DATASET_ROWS", len(sample_telco_df))
    monkeypatch.setattr(final_model, "EXPECTED_TRAINING_ROWS", 48)
    monkeypatch.setattr(final_model, "load_raw_data", lambda path: sample_telco_df.copy())
    monkeypatch.setattr(
        final_model,
        "get_imbalance_experiment_models",
        lambda: {"XGBoost (balanced)": LogisticRegression(max_iter=500)},
    )
    artifact_path = tmp_path / "models" / "customer_churn_model.joblib"
    metadata_path = tmp_path / "models" / "training_metadata.json"

    metadata = train_final_model(
        data_path=tmp_path / "synthetic.csv",
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )

    assert artifact_path.is_file() and artifact_path.stat().st_size > 0
    assert metadata_path.is_file()
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["split"]["training_rows"] == 48
    assert saved_metadata["split"]["holdout_labels_used_only_for_stratification"] is True
    assert saved_metadata["split"]["holdout_features_transformed"] is False
    assert saved_metadata["split"]["holdout_scored"] is False
    assert saved_metadata["provenance"]["historical_holdout_used_for_fitting"] is False
    assert saved_metadata["threshold_selection"]["metric"] == "f1"
    assert 0 <= saved_metadata["threshold_selection"]["threshold"] <= 1
    assert saved_metadata["artifact_size_bytes"] == artifact_path.stat().st_size
    loaded = load_model_artifact(artifact_path)
    assert loaded["threshold"] == pytest.approx(metadata["threshold_selection"]["threshold"])
