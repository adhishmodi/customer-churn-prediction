"""Validated single-customer and batch inference for the Phase 7A model artifact."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .calibration import CALIBRATION_METHODS, _apply_probability_calibrator
from .config import MODELS_DIR
from .preprocess import CATEGORICAL_FEATURES, FEATURE_COLUMNS

DEFAULT_ARTIFACT_PATH = MODELS_DIR / "customer_churn_model.joblib"
REQUIRED_ARTIFACT_KEYS = {
    "artifact_version",
    "model",
    "calibrator",
    "calibration_method",
    "threshold",
    "feature_columns",
    "metadata",
}


def prepare_inference_features(
    data: Mapping[str, Any] | pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """Validate and normalize inference rows without requiring a Churn target.

    A mapping represents one customer. A DataFrame represents a batch.
    Missing feature values are left for the fitted training imputers; unknown
    categorical strings are handled by the fitted OneHotEncoder.
    """
    if isinstance(data, Mapping):
        frame = pd.DataFrame([dict(data)])
        is_single = True
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
        is_single = False
    else:
        raise TypeError("data must be a feature mapping or pandas DataFrame.")

    if frame.empty:
        raise ValueError("At least one customer row is required.")
    if frame.columns.has_duplicates:
        raise ValueError("Input must not contain duplicate column names.")
    if not all(isinstance(column, str) for column in frame.columns):
        raise ValueError("Input column names must be strings.")

    supplied = set(frame.columns)
    required = set(FEATURE_COLUMNS)
    missing = sorted(required - supplied)
    extra = sorted(supplied - required)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing features: {missing}")
        if extra:
            details.append(f"unexpected columns: {extra}")
        raise ValueError("Invalid inference schema (" + "; ".join(details) + ").")

    frame = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in ("tenure", "MonthlyCharges"):
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{column} must contain numeric values or missing values."
            ) from exc
        if (frame[column].dropna() < 0).any():
            raise ValueError(f"{column} must not contain negative values.")

    # Match the deterministic conversion used when the training data was cleaned.
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    if (frame["TotalCharges"].dropna() < 0).any():
        raise ValueError("TotalCharges must not contain negative values.")
    zero_tenure_missing = frame["TotalCharges"].isna() & frame["tenure"].eq(0)
    frame.loc[zero_tenure_missing, "TotalCharges"] = 0.0

    senior = frame["SeniorCitizen"]
    allowed_senior_values = {0, 1, 0.0, 1.0, "0", "1", "No", "Yes"}
    invalid_senior = senior.notna() & ~senior.isin(allowed_senior_values)
    if invalid_senior.any():
        raise ValueError("SeniorCitizen must be one of 0, 1, 'No', or 'Yes'.")
    frame["SeniorCitizen"] = senior.replace(
        {0: "No", 1: "Yes", "0": "No", "1": "Yes"}
    )

    for column in CATEGORICAL_FEATURES:
        values = frame[column].dropna()
        if not values.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain categorical text values or missing values.")

    return frame, is_single


def _validate_artifact_bundle(bundle: Any) -> dict[str, Any]:
    """Validate the in-memory artifact contract before prediction."""
    if not isinstance(bundle, dict) or not REQUIRED_ARTIFACT_KEYS.issubset(bundle):
        raise ValueError("Model artifact is missing required bundle fields.")
    if type(bundle["artifact_version"]) is not int or bundle["artifact_version"] != 1:
        raise ValueError("Unsupported model artifact version.")
    if not isinstance(bundle["feature_columns"], (list, tuple)):
        raise ValueError("Model artifact feature schema must be a list or tuple.")
    if list(bundle["feature_columns"]) != list(FEATURE_COLUMNS):
        raise ValueError("Model artifact feature schema does not match this application.")
    if bundle["calibration_method"] not in CALIBRATION_METHODS:
        raise ValueError("Model artifact uses an unsupported calibration method.")

    threshold = bundle["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Model artifact threshold must be numeric.")
    if not np.isfinite(threshold) or not 0 <= float(threshold) <= 1:
        raise ValueError("Model artifact threshold must be finite and within [0, 1].")
    if not isinstance(bundle["metadata"], dict):
        raise ValueError("Model artifact metadata must be a dictionary.")

    model = bundle["model"]
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Model artifact does not contain a probability-capable model.")
    classifier = getattr(model, "named_steps", {}).get("classifier")
    classes = getattr(classifier, "classes_", None)
    if classes is None or 1 not in list(classes):
        raise ValueError("Model artifact classifier does not expose positive class 1.")
    if bundle["calibration_method"] != "none" and bundle["calibrator"] is None:
        raise ValueError("Model artifact is missing its fitted probability calibrator.")
    return bundle


def load_model_artifact(
    path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Load and validate a locally generated, trusted joblib model artifact.

    joblib uses pickle-based deserialization. Never load artifacts from an
    untrusted or unauthenticated source.
    """
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    return _validate_artifact_bundle(joblib.load(artifact_path))


def predict_customer(
    data: Mapping[str, Any] | pd.DataFrame,
    artifact: dict[str, Any] | str | Path,
) -> dict[str, Any] | pd.DataFrame:
    """Return calibrated churn probability, Yes/No label, and decision threshold.

    Mapping input returns one dictionary. DataFrame input returns a DataFrame
    with one result row per customer, preserving the input row order and index.
    """
    bundle = (
        load_model_artifact(artifact)
        if isinstance(artifact, (str, Path))
        else _validate_artifact_bundle(artifact)
    )
    features, is_single = prepare_inference_features(data)
    classifier = bundle["model"].named_steps["classifier"]
    classes = list(classifier.classes_)
    raw_probabilities = np.asarray(
        bundle["model"].predict_proba(features)[:, classes.index(1)], dtype=float
    )
    probabilities = _apply_probability_calibrator(
        bundle["calibration_method"], bundle["calibrator"], raw_probabilities
    )
    if (
        probabilities.ndim != 1
        or len(probabilities) != len(features)
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Model produced invalid calibrated churn probabilities.")

    threshold = float(bundle["threshold"])
    predictions = pd.DataFrame(
        {
            "churn_probability": probabilities,
            "predicted_churn": np.where(probabilities >= threshold, "Yes", "No"),
            "decision_threshold": threshold,
        },
        index=features.index,
    )
    return predictions.iloc[0].to_dict() if is_single else predictions
