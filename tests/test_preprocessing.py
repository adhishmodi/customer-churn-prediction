import numpy as np

from src.preprocess import (
    clean_data,
    fit_preprocessor,
    get_feature_names_out,
    prepare_features_target,
    split_data,
    transform_data,
)


def test_clean_data_handles_zero_tenure_without_global_imputation(sample_telco_df):
    sample_telco_df.loc[1, "tenure"] = 1
    cleaned = clean_data(sample_telco_df)

    assert "customerID" not in cleaned.columns
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert np.isnan(cleaned.loc[1, "TotalCharges"])
    assert set(cleaned["SeniorCitizen"].dropna().unique()) <= {"No", "Yes"}


def test_stratified_split_and_preprocessor_shapes(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_train) + len(X_test) == len(sample_telco_df)
    assert abs(y_train.mean() - y_test.mean()) < 0.03

    preprocessor = fit_preprocessor(X_train)
    train_values = transform_data(preprocessor, X_train)
    test_values = transform_data(preprocessor, X_test)
    feature_names = get_feature_names_out(preprocessor)

    assert train_values.shape[1] == len(feature_names)
    assert test_values.shape[1] == len(feature_names)
    assert np.isfinite(train_values).all()
    assert np.isfinite(test_values).all()


def test_unknown_category_is_ignored_without_changing_feature_count(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, X_test, _, _ = split_data(X, y)
    preprocessor = fit_preprocessor(X_train)

    expected_features = len(get_feature_names_out(preprocessor))
    X_test = X_test.copy()
    X_test.loc[X_test.index[0], "PaymentMethod"] = "New payment type"

    transformed = transform_data(preprocessor, X_test)
    assert transformed.shape[1] == expected_features



def test_clean_data_does_not_mutate_raw_dataframe(sample_telco_df):
    original = sample_telco_df.copy(deep=True)

    cleaned = clean_data(sample_telco_df)

    assert sample_telco_df.equals(original)
    assert "customerID" in sample_telco_df.columns
    assert sample_telco_df["TotalCharges"].dtype == object
    assert "customerID" not in cleaned.columns


def test_split_is_reproducible_and_stratified(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)

    first = split_data(X, y)
    second = split_data(X, y)

    for first_part, second_part in zip(first, second):
        assert first_part.equals(second_part)

    _, _, y_train, y_test = first
    assert abs(y_train.mean() - y_test.mean()) < 0.03


def test_preprocessor_fits_numeric_imputation_on_training_rows_only(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, _ = prepare_features_target(cleaned)
    X_train = X.iloc[:40].copy()
    X_test = X.iloc[40:].copy()

    X_train.loc[X_train.index[0], "MonthlyCharges"] = np.nan
    X_test.loc[:, "MonthlyCharges"] = 100_000.0

    preprocessor = fit_preprocessor(X_train)
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]

    expected_training_median = X_train["MonthlyCharges"].median()
    assert numeric_imputer.statistics_[1] == expected_training_median
    assert numeric_imputer.statistics_[1] < X_test["MonthlyCharges"].median()

    transformed_test = transform_data(preprocessor, X_test)
    assert np.isfinite(transformed_test).all()


def test_preprocessor_imputes_missing_numeric_and_categorical_values(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, _ = prepare_features_target(cleaned)
    X_train = X.iloc[:45].copy()
    X_test = X.iloc[45:].copy()

    X_train.loc[X_train.index[0], "MonthlyCharges"] = np.nan
    X_train.loc[X_train.index[1], "PaymentMethod"] = np.nan
    X_test.loc[X_test.index[0], "MonthlyCharges"] = np.nan
    X_test.loc[X_test.index[0], "PaymentMethod"] = np.nan

    preprocessor = fit_preprocessor(X_train)
    transformed_train = transform_data(preprocessor, X_train)
    transformed_test = transform_data(preprocessor, X_test)

    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_test).all()


def test_prepare_features_target_rejects_unexpected_target(sample_telco_df):
    import pytest

    sample_telco_df.loc[0, "Churn"] = "Maybe"
    cleaned = clean_data(sample_telco_df)

    with pytest.raises(ValueError, match="Unexpected or missing target values"):
        prepare_features_target(cleaned)
