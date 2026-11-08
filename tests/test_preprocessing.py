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
