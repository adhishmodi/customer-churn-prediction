"""Build the Phase 7A deployable model from the original training partition only."""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import MODELS_DIR, RANDOM_SEED, RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE
from .data_loader import load_raw_data
from .final_evaluation import fit_final_calibrated_model
from .preprocess import FEATURE_COLUMNS, clean_data, prepare_features_target
from .train import get_imbalance_experiment_models

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
EXPECTED_DATASET_ROWS = 7_043
EXPECTED_TRAINING_ROWS = 5_634
CALIBRATION_METHOD = "sigmoid"
MODEL_FILENAME = "customer_churn_model.joblib"
METADATA_FILENAME = "training_metadata.json"


def prepare_training_partition(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return only the original 80% training partition.

    The original stratified split is reconstructed on the raw rows. The
    non-training partition is discarded before cleaning or feature preparation,
    so no holdout feature is transformed or passed to a model function.
    """
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        raise ValueError("raw_df must be a non-empty pandas DataFrame.")
    if raw_df.columns.has_duplicates:
        raise ValueError("raw_df must not contain duplicate column names.")
    if TARGET_COLUMN not in raw_df.columns:
        raise ValueError(f"raw_df must contain the {TARGET_COLUMN!r} target column.")

    training_df, discarded_df = train_test_split(
        raw_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=raw_df[TARGET_COLUMN],
    )
    del discarded_df, raw_df

    cleaned_training = clean_data(training_df)
    X_train, y_train = prepare_features_target(cleaned_training)
    del training_df, cleaned_training

    if not X_train.index.equals(y_train.index):
        raise RuntimeError("Training feature and target indexes are misaligned.")
    if y_train.nunique() != 2:
        raise ValueError("The training partition must contain both churn classes.")
    return X_train, y_train


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def train_final_model(
    data_path: Path = RAW_DATA_PATH,
    artifact_path: Path = MODELS_DIR / MODEL_FILENAME,
    metadata_path: Path = MODELS_DIR / METADATA_FILENAME,
) -> dict[str, Any]:
    """Fit the Phase 4 balanced-XGBoost + sigmoid candidate and save its bundle.

    Only the original 5,634-row training partition is used for OOF calibration,
    F1-threshold selection, and final pipeline fitting. The historical holdout
    is not scored or used for model fitting or selection.
    """
    raw_df = load_raw_data(Path(data_path))
    if len(raw_df) != EXPECTED_DATASET_ROWS:
        raise ValueError(
            f"Expected the original {EXPECTED_DATASET_ROWS}-row Telco dataset; "
            f"received {len(raw_df)} rows. Refusing to create a mismatched artifact."
        )

    X_train, y_train = prepare_training_partition(raw_df)
    if len(X_train) != EXPECTED_TRAINING_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_ROWS} training rows from the original "
            f"split; received {len(X_train)}."
        )

    estimator = get_imbalance_experiment_models()["XGBoost (balanced)"]
    fitted = fit_final_calibrated_model(
        X_train,
        y_train,
        estimator,
        calibration_method=CALIBRATION_METHOD,
        cv_splits=5,
        random_state=RANDOM_SEED,
    )
    if fitted.get("test_set_used_for_fitting") is not False:
        raise RuntimeError("The fitted model did not confirm training-only provenance.")

    metadata: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "IBM Telco Customer Churn",
            "filename": Path(data_path).name,
            "rows": EXPECTED_DATASET_ROWS,
            "target": TARGET_COLUMN,
            "feature_count": len(FEATURE_COLUMNS),
        },
        "split": {
            "method": "stratified_train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_SEED,
            "training_rows": int(len(X_train)),
            "historical_holdout_rows": EXPECTED_DATASET_ROWS - len(X_train),
            "holdout_labels_used_only_for_stratification": True,
            "holdout_features_transformed": False,
            "holdout_features_used_for_fitting": False,
            "holdout_scored": False,
            "holdout_used_for_model_selection": False,
        },
        "model": {
            "candidate": "Phase 4 XGBoost (balanced)",
            "estimator_class": type(estimator).__name__,
            "parameters": estimator.get_params(deep=False),
        },
        "calibration": {
            "method": CALIBRATION_METHOD,
            "cv_splits": int(fitted["cv_splits"]),
            "random_state": int(fitted["random_state"]),
        },
        "threshold_selection": {
            "metric": "f1",
            "threshold": float(fitted["threshold"]),
            "training_oof_f1_at_selected_threshold": float(fitted["training_oof_f1"]),
            "interpretation": (
                "OOF F1 used to select the threshold; not an independent "
                "generalization estimate."
            ),
        },
        "provenance": {
            "training_rows_only": True,
            "historical_holdout_scored": False,
            "historical_holdout_used_for_fitting": False,
            "test_set_used_for_fitting": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": _installed_version("numpy"),
            "pandas": _installed_version("pandas"),
            "scikit_learn": _installed_version("scikit-learn"),
            "xgboost": _installed_version("xgboost"),
            "joblib": _installed_version("joblib"),
        },
        "feature_columns": list(FEATURE_COLUMNS),
    }

    bundle = {
        "artifact_version": ARTIFACT_VERSION,
        "model": fitted["model"],
        "calibrator": fitted["calibrator"],
        "calibration_method": fitted["calibration_method"],
        "threshold": float(fitted["threshold"]),
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata": metadata,
    }

    artifact_path = Path(artifact_path)
    metadata_path = Path(metadata_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path, compress=3)
    metadata["artifact_size_bytes"] = artifact_path.stat().st_size
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Saved Phase 7A model artifact (%d bytes) using %d training rows; "
        "selected threshold %.6f.",
        metadata["artifact_size_bytes"],
        len(X_train),
        fitted["threshold"],
    )
    # OOF arrays are intentionally not persisted in the deployable artifact.
    del fitted, X_train, y_train
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(json.dumps(train_final_model(), indent=2, sort_keys=True, default=str))
