"""Leakage-safe preprocessing for the Telco Customer Churn dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    ID_COLUMN,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .data_loader import load_raw_data

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, row-level cleaning without learning dataset statistics.

    Missing numeric and categorical values (other than the known zero-tenure
    TotalCharges case) are intentionally retained for fold-local imputers.
    """
    cleaned = df.copy()
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required - set(cleaned.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    zero_tenure_missing = cleaned["TotalCharges"].isna() & cleaned["tenure"].eq(0)
    cleaned.loc[zero_tenure_missing, "TotalCharges"] = 0.0

    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].replace(
        {0: "No", 1: "Yes", "0": "No", "1": "Yes"}
    )
    cleaned = cleaned.drop(columns=[ID_COLUMN], errors="ignore")
    return cleaned


def load_and_clean_data(path: Path | None = None) -> pd.DataFrame:
    """Load the original CSV and apply deterministic cleaning."""
    return clean_data(load_raw_data(path))


def get_preprocessing_pipeline() -> ColumnTransformer:
    """Create fresh numeric and categorical transformers for each model fit."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the fixed feature set from the binary churn target."""
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if y.isna().any():
        invalid = sorted(df.loc[y.isna(), TARGET_COLUMN].astype(str).unique().tolist())
        raise ValueError(f"Unexpected or missing target values: {invalid}")
    X = df[FEATURE_COLUMNS].copy()
    return X, y.astype(int)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/test split before fitting any transformer."""
    if y.nunique() != 2:
        raise ValueError("Churn training requires exactly two target classes.")
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit a preprocessing transformer on the supplied training rows only."""
    return get_preprocessing_pipeline().fit(X_train)


def transform_data(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Transform rows with an already-fitted training-only preprocessor."""
    return np.asarray(preprocessor.transform(X))


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Return transformed feature names in output-column order."""
    return preprocessor.get_feature_names_out().tolist()


def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path) -> ColumnTransformer:
    return joblib.load(path)


def save_processed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    output_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Save a fixed train/test preprocessing snapshot for inspection.

    Model selection must use the raw feature frames with a complete sklearn
    Pipeline, not these precomputed arrays, so each CV fold fits its own
    imputer, encoder, and scaler.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train.to_numpy())
    np.save(output_dir / "y_test.npy", y_test.to_numpy())
    (output_dir / "feature_names.txt").write_text(
        "\n".join(feature_names), encoding="utf-8"
    )


def run_preprocessing() -> dict[str, Any]:
    """Create the reproducible train/test preprocessing snapshot."""
    df = load_and_clean_data()
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = fit_preprocessor(X_train)
    X_train_processed = transform_data(preprocessor, X_train)
    X_test_processed = transform_data(preprocessor, X_test)
    feature_names = get_feature_names_out(preprocessor)

    save_preprocessor(preprocessor, PROCESSED_DATA_DIR / "preprocessor.joblib")
    save_processed_data(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        feature_names,
    )

    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train.to_numpy()
    )
    class_weights = dict(zip(classes.tolist(), weights.tolist()))
    counts = y_train.value_counts()
    scale_pos_weight = float(counts.loc[0] / counts.loc[1])

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "class_weights": class_weights,
        "scale_pos_weight": scale_pos_weight,
        "original_X_train": X_train,
        "original_X_test": X_test,
    }


if __name__ == "__main__":
    run_preprocessing()
