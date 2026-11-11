"""Leakage-resistant Phase 6 comparison of calibrated XGBoost finalists.

The public API accepts only the original training partition. It never loads,
receives, or evaluates the historical holdout.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from .calibration import (
    _apply_probability_calibrator,
    _fit_probability_calibrator,
    select_f1_threshold,
)
from .config import RANDOM_SEED
from .train import (
    build_model_pipeline,
    get_imbalance_experiment_models,
    get_tuning_distributions,
)

logger = logging.getLogger(__name__)

CANDIDATE_A = "Tuned XGBoost (unweighted)"
CANDIDATE_B = "Phase 4 XGBoost (balanced)"
CALIBRATION_METHOD = "sigmoid"


def _validate_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    outer_splits: int,
    calibration_splits: int,
    tuning_splits: int,
    n_iter: int,
) -> None:
    """Validate only the supplied training partition and resampling settings."""
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("X_train must be a DataFrame and y_train must be a Series.")
    if X_train.empty or len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must be non-empty and have the same number of rows.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train must have identical indexes and row order.")
    if X_train.columns.has_duplicates:
        raise ValueError("X_train must not contain duplicate feature columns.")
    if y_train.isna().any() or set(pd.unique(y_train)) != {0, 1}:
        raise ValueError("y_train must contain both binary labels 0 and 1 with no missing values.")
    if X_train.isna().all(axis=0).any():
        raise ValueError("X_train contains an entirely missing feature column.")
    for name, value in (
        ("outer_splits", outer_splits),
        ("calibration_splits", calibration_splits),
        ("tuning_splits", tuning_splits),
    ):
        if not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer of at least 2.")
    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be an integer of at least 1.")
    minority = int(y_train.value_counts().min())
    if minority < outer_splits:
        raise ValueError("The minority class must contain at least outer_splits observations.")
    outer_train_minority = minority - int(np.ceil(minority / outer_splits))
    if outer_train_minority < calibration_splits:
        raise ValueError(
            "Each outer-training partition must contain at least calibration_splits "
            "minority-class observations."
        )
    calibration_train_minority = outer_train_minority - int(
        np.ceil(outer_train_minority / calibration_splits)
    )
    if calibration_train_minority < tuning_splits:
        raise ValueError(
            "Each calibration-training partition must contain at least tuning_splits "
            "minority-class observations."
        )


def _search_unweighted_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator,
    distribution: dict[str, list[Any]],
    n_iter: int,
    tuning_splits: int,
    random_state: int,
) -> RandomizedSearchCV:
    """Tune unweighted XGBoost on only the rows passed to this helper."""
    cv = StratifiedKFold(
        n_splits=tuning_splits, shuffle=True, random_state=random_state
    )
    search = RandomizedSearchCV(
        estimator=build_model_pipeline(clone(estimator), feature_config="original"),
        param_distributions=distribution,
        n_iter=n_iter,
        scoring="roc_auc",
        refit=True,
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X, y)
    return search


def _positive_probability(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """Extract class-1 probabilities without assuming class-column ordering."""
    classes = list(estimator.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise ValueError("Fitted classifier does not expose positive class 1.")
    probabilities = np.asarray(estimator.predict_proba(X)[:, classes.index(1)], dtype=float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("Model produced invalid probability predictions.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model probabilities must lie in [0, 1].")
    return probabilities


def _nested_tuned_oof_probabilities(
    X_outer_train: pd.DataFrame,
    y_outer_train: pd.Series,
    estimator: BaseEstimator,
    distribution: dict[str, list[Any]],
    n_iter: int,
    calibration_splits: int,
    tuning_splits: int,
    random_state: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate OOF scores where each calibration fold is excluded from tuning."""
    calibration_cv = StratifiedKFold(
        n_splits=calibration_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof = np.full(len(X_outer_train), np.nan, dtype=float)
    nested_parameters: list[dict[str, Any]] = []

    for calibration_fold, (fit_idx, score_idx) in enumerate(
        calibration_cv.split(X_outer_train, y_outer_train), start=1
    ):
        X_fit = X_outer_train.iloc[fit_idx]
        y_fit = y_outer_train.iloc[fit_idx]
        X_score = X_outer_train.iloc[score_idx]
        fold_seed = random_state + calibration_fold
        search = _search_unweighted_xgboost(
            X_fit, y_fit, estimator, distribution, n_iter, tuning_splits, fold_seed
        )
        oof[score_idx] = _positive_probability(search.best_estimator_, X_score)
        nested_parameters.append(
            {
                "calibration_fold": calibration_fold,
                "best_params": dict(search.best_params_),
                "best_inner_roc_auc": float(search.best_score_),
            }
        )

    if not np.isfinite(oof).all():
        raise RuntimeError("Nested OOF generation did not predict every outer-training row.")
    return oof, nested_parameters


def _metric_row(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (calibrated_probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, calibrated_probabilities)),
        "average_precision": float(average_precision_score(y_true, calibrated_probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, calibrated_probabilities)),
        "log_loss": float(log_loss(y_true, calibrated_probabilities, labels=[0, 1])),
    }


def _summarize_results(
    fold_results: pd.DataFrame,
    fold_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Combine fold mean/std, pooled OOF metrics, and paired descriptive deltas."""
    rows: list[dict[str, Any]] = []
    metric_names = [
        "roc_auc", "average_precision", "precision", "recall", "f1",
        "accuracy", "brier_score", "log_loss",
    ]
    for candidate in (CANDIDATE_A, CANDIDATE_B):
        folds = fold_results.loc[fold_results["candidate"].eq(candidate)]
        predictions = fold_predictions.loc[fold_predictions["candidate"].eq(candidate)]
        y = predictions["y_true"].to_numpy(dtype=int)
        calibrated = predictions["calibrated_probability"].to_numpy(dtype=float)
        thresholded = predictions["predicted_churn"].to_numpy(dtype=int)
        row: dict[str, Any] = {
            "candidate": candidate,
            "fold_count": int(len(folds)),
            "pooled_roc_auc": float(roc_auc_score(y, calibrated)),
            "pooled_average_precision": float(average_precision_score(y, calibrated)),
            "pooled_precision": float(precision_score(y, thresholded, zero_division=0)),
            "pooled_recall": float(recall_score(y, thresholded, zero_division=0)),
            "pooled_f1": float(f1_score(y, thresholded, zero_division=0)),
            "pooled_accuracy": float(accuracy_score(y, thresholded)),
            "pooled_brier_score": float(brier_score_loss(y, calibrated)),
            "pooled_log_loss": float(log_loss(y, calibrated, labels=[0, 1])),
        }
        for metric in metric_names:
            row[f"fold_{metric}_mean"] = float(folds[metric].mean())
            row[f"fold_{metric}_std"] = float(folds[metric].std(ddof=1))
        row["mean_threshold"] = float(folds["threshold"].mean())
        row["std_threshold"] = float(folds["threshold"].std(ddof=1))
        rows.append(row)

    summary = pd.DataFrame(rows)
    a = summary.loc[summary["candidate"].eq(CANDIDATE_A)].iloc[0]
    b = summary.loc[summary["candidate"].eq(CANDIDATE_B)].iloc[0]
    delta: dict[str, Any] = {"candidate": f"Paired fold delta ({CANDIDATE_A} - {CANDIDATE_B})"}
    for metric in metric_names:
        paired = (
            fold_results.pivot(index="fold", columns="candidate", values=metric)[CANDIDATE_A]
            - fold_results.pivot(index="fold", columns="candidate", values=metric)[CANDIDATE_B]
        )
        delta[f"fold_{metric}_mean"] = float(paired.mean())
        delta[f"fold_{metric}_std"] = float(paired.std(ddof=1))
        delta[f"pooled_{metric}"] = float(a[f"pooled_{metric}"] - b[f"pooled_{metric}"])
    delta["fold_count"] = int(len(fold_results) // 2)
    delta["mean_threshold"] = float(a["mean_threshold"] - b["mean_threshold"])
    delta["std_threshold"] = np.nan
    return pd.concat([summary, pd.DataFrame([delta])], ignore_index=True)


def compare_calibrated_xgboost_finalists(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    outer_splits: int = 5,
    calibration_splits: int = 5,
    tuning_splits: int = 3,
    n_iter: int = 12,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Compare tuned unweighted and fixed balanced XGBoost using strict nested CV.

    Candidate A is tuned independently inside every calibration fold to produce
    leakage-resistant OOF probabilities. A second search tunes Candidate A on
    the complete outer-training fold before its outer-validation prediction.
    Candidate B uses the fixed Phase 4 balanced-XGBoost configuration. Both
    candidates fit a sigmoid calibrator and F1 threshold using outer-training
    OOF predictions only, then score the same untouched outer-validation rows.

    Only X_train and y_train are accepted; the historical holdout is excluded.
    """
    _validate_training_data(
        X_train, y_train, outer_splits, calibration_splits, tuning_splits, n_iter
    )
    candidates = get_imbalance_experiment_models()
    unweighted = candidates["XGBoost (unweighted)"]
    balanced = candidates["XGBoost (balanced)"]
    distribution = get_tuning_distributions()["XGBoost"]

    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []

    for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(
        outer_cv.split(X_train, y_train), start=1
    ):
        X_outer_train = X_train.iloc[outer_train_idx]
        y_outer_train = y_train.iloc[outer_train_idx]
        X_outer_valid = X_train.iloc[outer_valid_idx]
        y_outer_valid = y_train.iloc[outer_valid_idx]
        fold_seed = random_state + outer_fold * 100

        # Candidate A: every calibration OOF row is excluded from that fold's
        # hyperparameter search; no calibration-fold labels enter its tuning.
        a_oof_raw, nested_params = _nested_tuned_oof_probabilities(
            X_outer_train, y_outer_train, unweighted, distribution, n_iter,
            calibration_splits, tuning_splits, fold_seed,
        )
        a_final_search = _search_unweighted_xgboost(
            X_outer_train, y_outer_train, unweighted, distribution, n_iter,
            tuning_splits, fold_seed + 50,
        )
        a_outer_raw = _positive_probability(a_final_search.best_estimator_, X_outer_valid)
        a_calibrator = _fit_probability_calibrator(
            CALIBRATION_METHOD, a_oof_raw, y_outer_train.to_numpy(dtype=int)
        )
        a_oof_calibrated = _apply_probability_calibrator(
            CALIBRATION_METHOD, a_calibrator, a_oof_raw
        )
        a_threshold, a_training_f1 = select_f1_threshold(
            y_outer_train.to_numpy(dtype=int), a_oof_calibrated
        )
        a_outer_calibrated = _apply_probability_calibrator(
            CALIBRATION_METHOD, a_calibrator, a_outer_raw
        )

        # Candidate B: fixed Phase 4 estimator, with OOF probabilities generated
        # without fitting on the row being predicted.
        b_cv = StratifiedKFold(
            n_splits=calibration_splits,
            shuffle=True,
            random_state=fold_seed,
        )
        b_oof_matrix = cross_val_predict(
            build_model_pipeline(clone(balanced), feature_config="original"),
            X_outer_train, y_outer_train, cv=b_cv,
            method="predict_proba", n_jobs=1,
        )
        b_classes = list(np.unique(y_outer_train))
        b_oof_raw = b_oof_matrix[:, b_classes.index(1)]
        b_final_model = build_model_pipeline(clone(balanced), feature_config="original")
        b_final_model.fit(X_outer_train, y_outer_train)
        b_outer_raw = _positive_probability(b_final_model, X_outer_valid)
        b_calibrator = _fit_probability_calibrator(
            CALIBRATION_METHOD, b_oof_raw, y_outer_train.to_numpy(dtype=int)
        )
        b_oof_calibrated = _apply_probability_calibrator(
            CALIBRATION_METHOD, b_calibrator, b_oof_raw
        )
        b_threshold, b_training_f1 = select_f1_threshold(
            y_outer_train.to_numpy(dtype=int), b_oof_calibrated
        )
        b_outer_calibrated = _apply_probability_calibrator(
            CALIBRATION_METHOD, b_calibrator, b_outer_raw
        )

        for candidate, raw, calibrated, threshold, train_f1, params in (
            (CANDIDATE_A, a_outer_raw, a_outer_calibrated, a_threshold, a_training_f1,
             {"outer_best_params": dict(a_final_search.best_params_), "nested_oof_searches": nested_params}),
            (CANDIDATE_B, b_outer_raw, b_outer_calibrated, b_threshold, b_training_f1,
             {"fixed_configuration": balanced.get_params(deep=False)}),
        ):
            labels = y_outer_valid.to_numpy(dtype=int)
            metrics = _metric_row(labels, raw, calibrated, threshold)
            fold_rows.append({
                "candidate": candidate,
                "fold": outer_fold,
                "threshold": float(threshold),
                "training_oof_f1": float(train_f1),
                **metrics,
            })
            parameter_rows.append({
                "candidate": candidate,
                "fold": outer_fold,
                "threshold": float(threshold),
                "training_oof_f1": float(train_f1),
                "parameters_json": json.dumps(params, sort_keys=True, default=str),
            })
            for position, actual, raw_probability, calibrated_probability in zip(
                outer_valid_idx, labels, raw, calibrated
            ):
                prediction_rows.append({
                    "candidate": candidate,
                    "fold": outer_fold,
                    "sample_position": int(position),
                    "y_true": int(actual),
                    "raw_probability": float(raw_probability),
                    "calibrated_probability": float(calibrated_probability),
                    "threshold": float(threshold),
                    "predicted_churn": int(calibrated_probability >= threshold),
                })

        logger.info("Completed Phase 6 outer fold %d/%d", outer_fold, outer_splits)

    fold_results = pd.DataFrame(fold_rows)
    fold_predictions = pd.DataFrame(prediction_rows)
    thresholds_and_parameters = pd.DataFrame(parameter_rows)
    summary = _summarize_results(fold_results, fold_predictions)

    expected = len(X_train) * 2
    if len(fold_predictions) != expected:
        raise RuntimeError("Outer-fold predictions do not cover each training row exactly once per candidate.")
    for candidate in (CANDIDATE_A, CANDIDATE_B):
        counts = fold_predictions.loc[
            fold_predictions["candidate"].eq(candidate), "sample_position"
        ].value_counts()
        if len(counts) != len(X_train) or not counts.eq(1).all():
            raise RuntimeError(f"{candidate} predictions do not form a complete OOF partition.")

    return {
        "outer_fold_results": fold_results,
        "fold_predictions": fold_predictions,
        "summary": summary,
        "thresholds_and_parameters": thresholds_and_parameters,
        "configuration": {
            "outer_splits": outer_splits,
            "calibration_splits": calibration_splits,
            "tuning_splits": tuning_splits,
            "n_iter": n_iter,
            "random_state": random_state,
            "candidate_a": CANDIDATE_A,
            "candidate_b": CANDIDATE_B,
            "calibration_method": CALIBRATION_METHOD,
            "primary_metric": "f1",
            "candidate_a_nested_tuning": True,
            "test_set_used": False,
            "historical_holdout_accessed": False,
            "training_rows": int(len(X_train)),
        },
    }
