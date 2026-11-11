"""Training-only nested-CV experiments for probability calibration and thresholds."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import RANDOM_SEED
from .train import build_model_pipeline

CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")


def select_f1_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Choose the training-only threshold with maximum F1.

    Ties are resolved by choosing the threshold closest to 0.50, then the
    lower threshold. The function never receives or infers a test partition.
    """
    from sklearn.metrics import precision_recall_curve

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("y_true and probabilities must be aligned 1D arrays.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both binary classes.")

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("At least two distinct probability values are required.")
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_f1 = float(np.max(f1_values))
    tied = np.flatnonzero(np.isclose(f1_values, best_f1))
    best_index = min(
        tied,
        key=lambda index: (abs(float(thresholds[index]) - 0.5), float(thresholds[index])),
    )
    return float(thresholds[best_index]), best_f1


def _fit_probability_calibrator(
    method: str,
    probabilities: np.ndarray,
    y_true: pd.Series | np.ndarray,
) -> Any:
    """Fit a calibration mapping on out-of-fold training probabilities only."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unsupported calibration method: {method}")
    scores = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError("Calibration probabilities and labels must be aligned.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Calibration probabilities must be finite values in [0, 1].")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration requires both binary classes.")

    if method == "none":
        return None
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(scores, labels)

    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return LogisticRegression(solver="lbfgs", random_state=RANDOM_SEED).fit(
        logits, labels
    )


def _apply_probability_calibrator(method: str, calibrator: Any, probabilities: np.ndarray) -> np.ndarray:
    """Apply a fitted calibration mapping while preserving [0, 1] bounds."""
    scores = np.asarray(probabilities, dtype=float)
    if method == "none":
        return np.clip(scores, 0.0, 1.0)
    if method == "isotonic":
        calibrated = calibrator.predict(scores)
    elif method == "sigmoid":
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibrated = calibrator.predict_proba(logits)[:, 1]
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
    return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


def evaluate_calibration_threshold_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, BaseEstimator],
    outer_splits: int = 5,
    inner_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Evaluate calibration + F1-threshold selection with nested stratified CV.

    For each outer fold, inner out-of-fold probabilities from the outer-training
    partition fit the calibrator and select its F1 threshold. A fresh base model
    is then fitted on the complete outer-training partition and evaluated once
    on the untouched outer-validation fold. The held-out project test partition
    is not accepted by this API.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if y_train.nunique() != 2:
        raise ValueError("Calibration CV requires exactly two target classes.")
    if not models:
        raise ValueError("At least one candidate model must be supplied.")
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must both be at least 2.")
    if int(y_train.value_counts().min()) < outer_splits:
        raise ValueError("The minority class must contain at least outer_splits rows.")

    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for model_name, estimator in models.items():
        for fold_number, (outer_fit_idx, outer_valid_idx) in enumerate(
            outer_cv.split(X_train, y_train), start=1
        ):
            X_outer_fit = X_train.iloc[outer_fit_idx]
            y_outer_fit = y_train.iloc[outer_fit_idx]
            X_outer_valid = X_train.iloc[outer_valid_idx]
            y_outer_valid = y_train.iloc[outer_valid_idx]
            if int(y_outer_fit.value_counts().min()) < inner_splits:
                raise ValueError(
                    "The outer-training minority class must contain at least "
                    "inner_splits rows."
                )

            inner_cv = StratifiedKFold(
                n_splits=inner_splits,
                shuffle=True,
                random_state=random_state + fold_number,
            )
            inner_oof = cross_val_predict(
                build_model_pipeline(clone(estimator)),
                X_outer_fit,
                y_outer_fit,
                cv=inner_cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]

            fitted_model = build_model_pipeline(clone(estimator))
            fitted_model.fit(X_outer_fit, y_outer_fit)
            outer_raw = fitted_model.predict_proba(X_outer_valid)[:, 1]

            for method in CALIBRATION_METHODS:
                calibrator = _fit_probability_calibrator(
                    method, inner_oof, y_outer_fit
                )
                inner_calibrated = _apply_probability_calibrator(
                    method, calibrator, inner_oof
                )
                threshold, training_f1 = select_f1_threshold(
                    y_outer_fit, inner_calibrated
                )
                outer_calibrated = _apply_probability_calibrator(
                    method, calibrator, outer_raw
                )
                predictions = (outer_calibrated >= threshold).astype(int)
                labels = y_outer_valid.to_numpy(dtype=int)

                metrics = {
                    "roc_auc": float(roc_auc_score(labels, outer_calibrated)),
                    "average_precision": float(
                        average_precision_score(labels, outer_calibrated)
                    ),
                    "precision": float(
                        precision_score(labels, predictions, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(labels, predictions, zero_division=0)
                    ),
                    "f1": float(f1_score(labels, predictions, zero_division=0)),
                    "accuracy": float(accuracy_score(labels, predictions)),
                    "brier_score": float(
                        brier_score_loss(labels, outer_calibrated)
                    ),
                    "log_loss": float(
                        log_loss(labels, outer_calibrated, labels=[0, 1])
                    ),
                }
                fold_rows.append(
                    {
                        "model": model_name,
                        "calibration": method,
                        "fold": fold_number,
                        "threshold": threshold,
                        "training_oof_f1": training_f1,
                        **metrics,
                    }
                )
                for row_index, (actual, raw, calibrated) in enumerate(
                    zip(labels, outer_raw, outer_calibrated)
                ):
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "calibration": method,
                            "fold": fold_number,
                            "y_true": int(actual),
                            "raw_probability": float(raw),
                            "calibrated_probability": float(calibrated),
                            "threshold": threshold,
                        }
                    )

    fold_scores = pd.DataFrame(fold_rows)
    metric_columns = [
        "roc_auc", "average_precision", "precision", "recall", "f1",
        "accuracy", "brier_score", "log_loss", "threshold", "training_oof_f1",
    ]
    summary = (
        fold_scores.groupby(["model", "calibration"], sort=False)[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    return {
        "fold_scores": fold_scores,
        "summary": summary,
        "validation_predictions": pd.DataFrame(prediction_rows),
        "configuration": {
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "random_state": random_state,
            "threshold_metric": "f1",
            "calibration_methods": list(CALIBRATION_METHODS),
            "training_rows": len(X_train),
            "test_set_used": False,
        },
    }
