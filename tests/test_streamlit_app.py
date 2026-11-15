import numpy as np
import pandas as pd
import pytest

import app


def test_parse_optional_number_accepts_number_and_blank():
    assert app.parse_optional_number("12.5", "tenure") == 12.5
    assert np.isnan(app.parse_optional_number(" ", "tenure"))


@pytest.mark.parametrize("value", ["abc", "-1", "-0.01", "inf", "-inf"])
def test_parse_optional_number_rejects_invalid_or_negative(value):
    with pytest.raises(ValueError):
        app.parse_optional_number(value, "tenure")


def test_validate_upload_size_accepts_limit_and_rejects_oversize():
    assert app.validate_upload_size(b"1234", max_bytes=4) == b"1234"
    with pytest.raises(ValueError, match="no larger"):
        app.validate_upload_size(b"12345", max_bytes=4)


def test_validate_batch_frame_preserves_exact_feature_schema(sample_telco_df):
    source = sample_telco_df.drop(columns=["customerID", "Churn"]).iloc[:3].copy()
    normalized = app.validate_batch_frame(source)

    assert list(normalized.columns) == app.FEATURE_COLUMNS
    assert len(normalized) == 3
    assert normalized.index.equals(source.index)


def test_validate_batch_frame_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one"):
        app.validate_batch_frame(pd.DataFrame())


def test_validate_batch_frame_rejects_missing_and_extra_columns(sample_telco_df):
    source = sample_telco_df.drop(columns=["customerID", "Churn"]).iloc[:2].copy()
    with pytest.raises(ValueError, match="missing features"):
        app.validate_batch_frame(source.drop(columns=["tenure"]))
    with pytest.raises(ValueError, match="unexpected columns"):
        app.validate_batch_frame(source.assign(Churn="Yes"))


def test_validate_batch_frame_rejects_duplicate_columns(sample_telco_df):
    source = sample_telco_df.drop(columns=["customerID", "Churn"]).iloc[:2].copy()
    source.columns = list(source.columns[:-1]) + [source.columns[0]]
    with pytest.raises(ValueError, match="duplicate"):
        app.validate_batch_frame(source)


def test_validate_batch_frame_enforces_row_limit(sample_telco_df):
    source = sample_telco_df.drop(columns=["customerID", "Churn"]).iloc[:3].copy()
    with pytest.raises(ValueError, match="limited"):
        app.validate_batch_frame(source, max_rows=2)


def test_predict_batch_preserves_order_and_appends_predictions(sample_telco_df, monkeypatch):
    source = sample_telco_df.drop(columns=["customerID", "Churn"]).iloc[[5, 2, 9]].copy()

    def fake_predict(features, artifact):
        return pd.DataFrame(
            {
                "churn_probability": [0.2, 0.8, 0.4],
                "predicted_churn": ["No", "Yes", "Yes"],
                "decision_threshold": [0.35, 0.35, 0.35],
            },
            index=features.index,
        )

    monkeypatch.setattr(app, "predict_customer", fake_predict)
    result = app.predict_batch(source, artifact={})

    assert len(result) == 3
    assert result["churn_probability"].tolist() == [0.2, 0.8, 0.4]
    assert result["predicted_churn"].tolist() == ["No", "Yes", "Yes"]
    assert result["tenure"].tolist() == source["tenure"].tolist()
    assert result.index.tolist() == [0, 1, 2]


def test_make_download_csv_neutralizes_formula_like_text():
    source = pd.DataFrame({"category": ["=1+1", "+cmd", "-x", "@value", " =1+1", "safe"]})
    exported = app.make_download_csv(source).decode("utf-8")

    assert "'=1+1" in exported
    assert "'+cmd" in exported
    assert "'-x" in exported
    assert "'@value" in exported
    assert "' =1+1" in exported
    assert "safe" in exported
