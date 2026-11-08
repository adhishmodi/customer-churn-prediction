import numpy as np
from sklearn.linear_model import LogisticRegression

from src.preprocess import clean_data, prepare_features_target, split_data
from src.train import (
    build_model_pipeline,
    calculate_class_weights,
    cross_validate_model,
    get_best_model,
    FoldBalancedXGBClassifier,
)


def test_class_weights_are_derived_from_training_labels():
    labels = __import__("pandas").Series([0] * 45 + [1] * 15)
    weights, scale_pos_weight = calculate_class_weights(labels)

    assert np.isclose(weights[0], 60 / (2 * 45))
    assert np.isclose(weights[1], 60 / (2 * 15))
    assert np.isclose(scale_pos_weight, 3.0)


def test_cross_validation_runs_with_preprocessing_inside_pipeline(sample_telco_df):
    cleaned = clean_data(sample_telco_df)
    X, y = prepare_features_target(cleaned)
    X_train, _, y_train, _ = split_data(X, y)
    model = build_model_pipeline(LogisticRegression(max_iter=500))

    metrics = cross_validate_model(model, X_train, y_train)

    assert set(metrics) == {
        "cv_roc_auc_mean", "cv_roc_auc_std",
        "cv_precision_mean", "cv_precision_std",
        "cv_recall_mean", "cv_recall_std",
        "cv_f1_mean", "cv_f1_std",
    }
    assert 0.0 <= metrics["cv_roc_auc_mean"] <= 1.0


def test_best_model_uses_cross_validation_not_test_metrics():
    results = {
        "model_A": {
            "model": object(),
            "cv_metrics": {"cv_roc_auc_mean": 0.82},
            "test_metrics": {"roc_auc": 0.50},
        },
        "model_B": {
            "model": object(),
            "cv_metrics": {"cv_roc_auc_mean": 0.74},
            "test_metrics": {"roc_auc": 0.99},
        },
    }
    name, _ = get_best_model(results)
    assert name == "model_A"



def test_xgboost_class_ratio_is_computed_from_each_fit():
    features = np.arange(20, dtype=float).reshape(-1, 1)
    labels = np.array([0] * 15 + [1] * 5)
    classifier = FoldBalancedXGBClassifier(n_estimators=5, max_depth=2)
    classifier.fit(features, labels)

    assert classifier.scale_pos_weight_ == 3.0
    assert classifier.predict_proba(features).shape == (20, 2)
