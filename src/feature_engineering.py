"""Leakage-safe, deterministic feature engineering for churn experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

FEATURE_CONFIGS = {
    "original": {"tenure_groups": False, "service_count": False},
    "tenure_groups": {"tenure_groups": True, "service_count": False},
    "service_count": {"tenure_groups": False, "service_count": True},
    "tenure_groups_and_service_count": {
        "tenure_groups": True,
        "service_count": True,
    },
}

SERVICE_INDICATORS = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Append optional customer-tenure bands and subscribed-service count.

    The transformations are deterministic and learn no dataset statistics.
    They are nevertheless placed inside the model Pipeline so every
    experiment has one consistent, fold-local preprocessing boundary.
    """

    def __init__(self, feature_config: str = "original") -> None:
        self.feature_config = feature_config

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureEngineeringTransformer":
        if self.feature_config not in FEATURE_CONFIGS:
            raise ValueError(
                f"Unknown feature configuration: {self.feature_config!r}. "
                f"Choose from {sorted(FEATURE_CONFIGS)}."
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Feature engineering expects a pandas DataFrame.")
        required = set()
        options = FEATURE_CONFIGS[self.feature_config]
        if options["tenure_groups"]:
            required.add("tenure")
        if options["service_count"]:
            required.update(SERVICE_INDICATORS)
        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(f"Missing columns for feature engineering: {missing}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Feature engineering expects a pandas DataFrame.")
        missing_original = sorted(set(self.feature_names_in_) - set(X.columns))
        if missing_original:
            raise ValueError(f"Input is missing fitted columns: {missing_original}")

        transformed = X.copy()
        options = FEATURE_CONFIGS[self.feature_config]
        if options["tenure_groups"]:
            tenure = pd.to_numeric(transformed["tenure"], errors="coerce")
            bands = pd.cut(
                tenure,
                bins=[-np.inf, 12, 24, 48, 60, np.inf],
                labels=[
                    "0-12 months",
                    "13-24 months",
                    "25-48 months",
                    "49-60 months",
                    "61+ months",
                ],
                right=True,
            )
            transformed["tenure_group"] = bands.astype("object").where(
                bands.notna(), "Unknown"
            )

        if options["service_count"]:
            subscribed = pd.DataFrame(
                {
                    column: transformed[column].eq("Yes")
                    for column in SERVICE_INDICATORS
                },
                index=transformed.index,
            )
            transformed["service_count"] = subscribed.sum(axis=1).astype("int64")

        return transformed

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        names = list(self.feature_names_in_ if input_features is None else input_features)
        options = FEATURE_CONFIGS[self.feature_config]
        if options["tenure_groups"]:
            names.append("tenure_group")
        if options["service_count"]:
            names.append("service_count")
        return np.asarray(names, dtype=object)
