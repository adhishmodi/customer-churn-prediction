import pandas as pd
import pytest

from src.validate import DataValidationError, validate_dataset


def test_validation_accepts_valid_dataset(sample_telco_df):
    result = validate_dataset(sample_telco_df)
    assert result["passed"] is True
    assert result["info"]["unique_ids"] == len(sample_telco_df)


def test_validation_rejects_missing_required_columns():
    with pytest.raises(DataValidationError):
        validate_dataset(pd.DataFrame({"Churn": ["Yes", "No"]}))
