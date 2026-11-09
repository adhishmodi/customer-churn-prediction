"""Repeated cross-validation experiments for model stability assessment."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from .config import RANDOM_SEED
from .train import SCORING, build_model_pipeline, get_models

METRIC_NAMES = tuple(SCORING)


def get_repeated_cv_strategy(
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = RANDOM_SEED,
) -> RepeatedStratifiedKFold:
    """Create a reproducible repeated stratified cross-validation strategy."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )


def compare_models_repeated_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, BaseEstimator] | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Compare fixed model configurations using repeated CV on training rows only.

    Fold standard deviations are descriptive variability, not formal confidence
    intervals: repeated CV folds are correlated because their training sets overlap.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if y_train.nunique() != 2:
        raise ValueError("Repeated stratified CV requires exactly two target classes.")

    candidates = get_models() if models is None else models
    if not candidates:
        raise ValueError("At least one candidate model must be supplied.")

    minority_count = int(y_train.value_counts().min())
    if minority_count < n_splits:
        raise ValueError(
            "The minority class must contain at least n_splits observations "
            f"(found {minority_count}, requested {n_splits})."
        )

    cv = get_repeated_cv_strategy(n_splits, n_repeats, random_state)
    rows: list[dict[str, Any]] = []
    total_splits = n_splits * n_repeats

    for model_name, estimator in candidates.items():
        scores = cross_validate(
            build_model_pipeline(estimator),
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        for split_index in range(total_splits):
            row: dict[str, Any] = {
                "model": model_name,
                "repeat": split_index // n_splits + 1,
                "fold": split_index % n_splits + 1,
                "fit_time": float(scores["fit_time"][split_index]),
                "score_time": float(scores["score_time"][split_index]),
            }
            for metric_name in METRIC_NAMES:
                row[metric_name] = float(scores[f"test_{metric_name}"][split_index])
            rows.append(row)

    fold_scores = pd.DataFrame(rows)
    metric_columns = list(METRIC_NAMES) + ["fit_time", "score_time"]
    grouped = fold_scores.groupby("model", sort=False)[metric_columns].agg(
        ["mean", "std", "median", "min", "max"]
    )
    grouped.columns = [
        f"{metric}_{statistic}" for metric, statistic in grouped.columns
    ]
    summary = grouped.reset_index()
    summary.insert(1, "fold_count", total_splits)

    repeat_means = (
        fold_scores.groupby(["repeat", "model"], as_index=False)["roc_auc"]
        .mean()
        .rename(columns={"roc_auc": "mean_roc_auc"})
    )
    winner_indices = repeat_means.groupby("repeat")["mean_roc_auc"].idxmax()
    repeat_winners = (
        repeat_means.loc[winner_indices]
        .sort_values("repeat")
        .rename(columns={"model": "selected_model"})
        .reset_index(drop=True)
    )

    wins = repeat_winners["selected_model"].value_counts()
    selection_stability = pd.DataFrame(
        {
            "model": list(candidates),
            "winning_repeats": [
                int(wins.get(name, 0)) for name in candidates
            ],
        }
    )
    selection_stability["selection_frequency"] = (
        selection_stability["winning_repeats"] / n_repeats
    )
    selection_stability = selection_stability.sort_values(
        ["winning_repeats", "model"], ascending=[False, True]
    ).reset_index(drop=True)

    return {
        "fold_scores": fold_scores,
        "summary": summary,
        "repeat_winners": repeat_winners,
        "selection_stability": selection_stability,
        "configuration": {
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "random_state": random_state,
            "selection_metric": "roc_auc",
            "training_rows": len(X_train),
            "test_set_used": False,
        },
    }
