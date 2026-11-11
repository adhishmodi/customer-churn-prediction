"""Training-only finalist selection and frozen final-model evaluation helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .calibration import (
    _apply_probability_calibrator, _fit_probability_calibrator,
    select_f1_threshold,
)
from .config import RANDOM_SEED
from .train import build_model_pipeline

PHASE3_FINALISTS = ("XGBoost (balanced)", "XGBoost (unweighted)")
FINAL_CALIBRATION_METHOD = "sigmoid"


def select_phase3_finalist(fold_scores: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Select XGBoost finalist from Phase 3 sigmoid outer folds; test is not accepted."""
    required = {"model", "calibration", "fold", "f1", "precision", "recall"}
    missing = sorted(required - set(fold_scores.columns))
    if missing:
        raise ValueError(f"Phase 3 fold scores are missing columns: {missing}")

    rows = fold_scores.loc[
        fold_scores["model"].isin(PHASE3_FINALISTS)
        & fold_scores["calibration"].eq(FINAL_CALIBRATION_METHOD),
        ["model", "calibration", "fold", "f1", "precision", "recall"],
    ].copy()
    if rows.empty:
        raise ValueError("No Phase 3 sigmoid results found for the XGBoost finalists.")

    rows["fold"] = pd.to_numeric(rows["fold"], errors="raise").astype(int)
    for metric in ("f1", "precision", "recall"):
        rows[metric] = pd.to_numeric(rows[metric], errors="raise")
    values = rows[["f1", "precision", "recall"]].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Phase 3 candidate metrics must be finite and within [0, 1].")

    summaries = []
    for name in PHASE3_FINALISTS:
        candidate = rows.loc[rows["model"].eq(name)]
        if set(candidate["fold"]) != {1, 2, 3, 4, 5} or candidate["fold"].duplicated().any():
            raise ValueError(f"{name} must have exactly one sigmoid result for each of five outer folds.")
        summaries.append({
            "model": name,
            "calibration": FINAL_CALIBRATION_METHOD,
            "f1_mean": float(candidate["f1"].mean()),
            "f1_std": float(candidate["f1"].std(ddof=1)),
            "precision_mean": float(candidate["precision"].mean()),
            "recall_mean": float(candidate["recall"].mean()),
        })

    summary = pd.DataFrame(summaries).sort_values(
        ["f1_mean", "precision_mean", "recall_mean", "model"],
        ascending=[False, False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    return str(summary.loc[0, "model"]), summary


def fit_final_calibrated_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimator: BaseEstimator,
    calibration_method: str = FINAL_CALIBRATION_METHOD,
    cv_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Fit calibrator/threshold from training-only OOF scores, then refit all train rows."""
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if y_train.nunique() != 2:
        raise ValueError("Final fitting requires exactly two target classes.")
    if cv_splits < 2 or int(y_train.value_counts().min()) < cv_splits:
        raise ValueError("Insufficient minority-class rows for the requested CV folds.")
    if calibration_method not in ("none", "sigmoid", "isotonic"):
        raise ValueError(f"Unsupported calibration method: {calibration_method}")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    oof_raw = cross_val_predict(
        build_model_pipeline(clone(estimator)), X_train, y_train,
        cv=cv, method="predict_proba", n_jobs=1,
    )[:, 1]
    calibrator = _fit_probability_calibrator(
        calibration_method, oof_raw, y_train.to_numpy(dtype=int)
    )
    oof_calibrated = _apply_probability_calibrator(
        calibration_method, calibrator, oof_raw
    )
    threshold, training_oof_f1 = select_f1_threshold(
        y_train.to_numpy(dtype=int), oof_calibrated
    )

    final_pipeline = build_model_pipeline(clone(estimator))
    final_pipeline.fit(X_train, y_train)
    return {
        "model": final_pipeline,
        "calibrator": calibrator,
        "calibration_method": calibration_method,
        "threshold": float(threshold),
        "training_oof_f1": float(training_oof_f1),
        "oof_raw_probabilities": oof_raw,
        "oof_calibrated_probabilities": oof_calibrated,
        "training_rows": int(len(X_train)),
        "cv_splits": int(cv_splits),
        "random_state": int(random_state),
        "test_set_used_for_fitting": False,
    }


def predict_final_probabilities(fitted: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Return calibrated probabilities from the frozen final-model bundle."""
    if fitted.get("test_set_used_for_fitting") is not False:
        raise ValueError("Final model bundle has an invalid fitting provenance marker.")
    model = fitted["model"]
    classifier = model.named_steps["classifier"]
    class_index = list(classifier.classes_).index(1)
    raw = model.predict_proba(X)[:, class_index]
    calibrated = _apply_probability_calibrator(
        fitted["calibration_method"], fitted["calibrator"], raw
    )
    if not np.isfinite(calibrated).all() or ((calibrated < 0) | (calibrated > 1)).any():
        raise ValueError("Final calibrated probabilities must be finite and within [0, 1].")
    return calibrated


def evaluate_final_model(
    fitted: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen model at 0.50 and the training-selected threshold; never tune on test."""
    if len(X_test) != len(y_test) or y_test.nunique() != 2:
        raise ValueError("Test features and binary labels must be aligned.")
    probabilities = predict_final_probabilities(fitted, X_test)
    labels = y_test.to_numpy(dtype=int)
    metric_rows, prediction_rows = [], []
    for strategy, threshold in (
        ("default_0.50", 0.50), ("training_oof_f1", fitted["threshold"])
    ):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        metric_rows.append({
            "threshold_strategy": strategy, "threshold": float(threshold),
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "brier_score": float(brier_score_loss(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        })
        for actual, probability, prediction in zip(labels, probabilities, predictions):
            prediction_rows.append({
                "actual_churn": int(actual),
                "calibrated_probability": float(probability),
                "threshold_strategy": strategy,
                "threshold": float(threshold),
                "predicted_churn": int(prediction),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)
