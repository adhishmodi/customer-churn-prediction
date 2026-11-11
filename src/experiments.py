"""Repeated cross-validation experiments for model stability assessment."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from .config import RANDOM_SEED
from .train import SCORING, build_model_pipeline, get_models
from .feature_engineering import FEATURE_CONFIGS

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


def evaluate_nested_cv_model_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_names: list[str] | None = None,
    outer_splits: int = 3,
    inner_splits: int = 3,
    n_iter: int = 5,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Estimate a complete model-and-hyperparameter selection procedure.

    Each outer fold is held out from both inner hyperparameter tuning and model
    selection. Within the outer training portion, each requested model family
    is tuned by inner stratified CV; the highest inner-CV ROC-AUC candidate is
    evaluated once on the outer validation fold.

    The function accepts training data only. Its outer-fold metrics estimate
    the selection procedure, not the final refitted model's test performance.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    from .train import get_tuning_distributions

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if y_train.nunique() != 2:
        raise ValueError("Nested CV requires exactly two target classes.")
    if outer_splits < 2:
        raise ValueError("outer_splits must be at least 2.")
    if inner_splits < 2:
        raise ValueError("inner_splits must be at least 2.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    estimators = get_models()
    distributions = get_tuning_distributions()
    selected_names = list(distributions) if model_names is None else model_names
    if not selected_names:
        raise ValueError("At least one tunable model must be selected.")
    unknown = sorted(set(selected_names) - set(distributions))
    if unknown:
        raise ValueError(f"Models do not have tuning spaces: {unknown}")

    if int(y_train.value_counts().min()) < outer_splits:
        raise ValueError(
            "The minority class must contain at least outer_splits observations."
        )

    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    outer_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(
        outer_cv.split(X_train, y_train), start=1
    ):
        X_outer_train = X_train.iloc[outer_train_idx]
        y_outer_train = y_train.iloc[outer_train_idx]
        X_outer_valid = X_train.iloc[outer_valid_idx]
        y_outer_valid = y_train.iloc[outer_valid_idx]

        if int(y_outer_train.value_counts().min()) < inner_splits:
            raise ValueError(
                "Each outer training fold must contain at least inner_splits "
                "examples from the minority class."
            )

        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=random_state + outer_fold,
        )
        searches: dict[str, Any] = {}
        for model_name in selected_names:
            search = RandomizedSearchCV(
                estimator=build_model_pipeline(estimators[model_name]),
                param_distributions=distributions[model_name],
                n_iter=n_iter,
                scoring=SCORING,
                refit="roc_auc",
                cv=inner_cv,
                random_state=random_state + outer_fold,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_outer_train, y_outer_train)
            searches[model_name] = search
            best_index = int(search.best_index_)
            candidate_rows.append(
                {
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "inner_cv_roc_auc_mean": float(search.best_score_),
                    "inner_cv_roc_auc_std": float(
                        search.cv_results_["std_test_roc_auc"][best_index]
                    ),
                    "best_params": dict(search.best_params_),
                }
            )

        selected_name = max(
            selected_names, key=lambda name: searches[name].best_score_
        )
        selected_search = searches[selected_name]
        selected_pipeline = selected_search.best_estimator_
        classifier = selected_pipeline.named_steps["classifier"]
        class_index = list(classifier.classes_).index(1)
        probabilities = selected_pipeline.predict_proba(X_outer_valid)[:, class_index]
        predictions = selected_pipeline.predict(X_outer_valid)

        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_model": selected_name,
                "inner_cv_roc_auc": float(selected_search.best_score_),
                "outer_roc_auc": float(roc_auc_score(y_outer_valid, probabilities)),
                "outer_average_precision": float(
                    average_precision_score(y_outer_valid, probabilities)
                ),
                "outer_precision": float(
                    precision_score(y_outer_valid, predictions, zero_division=0)
                ),
                "outer_recall": float(
                    recall_score(y_outer_valid, predictions, zero_division=0)
                ),
                "outer_f1": float(
                    f1_score(y_outer_valid, predictions, zero_division=0)
                ),
                "outer_accuracy": float(
                    accuracy_score(y_outer_valid, predictions)
                ),
                "best_params": dict(selected_search.best_params_),
            }
        )

    outer_results = pd.DataFrame(outer_rows)
    candidate_results = pd.DataFrame(candidate_rows)
    outer_metric_columns = [
        "outer_roc_auc",
        "outer_average_precision",
        "outer_precision",
        "outer_recall",
        "outer_f1",
        "outer_accuracy",
    ]
    summary_rows = []
    for metric in outer_metric_columns:
        values = outer_results[metric]
        summary_rows.append(
            {
                "metric": metric.removeprefix("outer_"),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    outer_summary = pd.DataFrame(summary_rows)

    selected_counts = outer_results["selected_model"].value_counts()
    selection_frequency = pd.DataFrame(
        {
            "model": selected_names,
            "selected_outer_folds": [
                int(selected_counts.get(name, 0)) for name in selected_names
            ],
        }
    )
    selection_frequency["selection_frequency"] = (
        selection_frequency["selected_outer_folds"] / outer_splits
    )

    return {
        "outer_fold_results": outer_results,
        "inner_candidate_results": candidate_results,
        "outer_summary": outer_summary,
        "selection_frequency": selection_frequency.sort_values(
            ["selected_outer_folds", "model"], ascending=[False, True]
        ).reset_index(drop=True),
        "configuration": {
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "n_iter": n_iter,
            "random_state": random_state,
            "selection_metric": "roc_auc",
            "training_rows": len(X_train),
            "test_set_used": False,
            "selection_procedure_evaluated": True,
        },
    }



def compare_feature_engineering_repeated_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, BaseEstimator],
    feature_configs: list[str] | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Compare model/feature combinations using identical training-only CV folds.

    All feature transformations and learned preprocessing remain inside each
    estimator Pipeline. The API accepts only the training partition.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if y_train.nunique() != 2:
        raise ValueError("Feature-engineering CV requires exactly two target classes.")
    if not models:
        raise ValueError("At least one candidate model must be supplied.")

    selected_configs = list(FEATURE_CONFIGS) if feature_configs is None else feature_configs
    if not selected_configs:
        raise ValueError("At least one feature configuration must be supplied.")
    unknown = sorted(set(selected_configs) - set(FEATURE_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown feature configurations: {unknown}")
    if len(set(selected_configs)) != len(selected_configs):
        raise ValueError("Feature configurations must not contain duplicates.")
    if len(set(models)) != len(models):
        raise ValueError("Model names must be unique.")

    minority_count = int(y_train.value_counts().min())
    if minority_count < n_splits:
        raise ValueError(
            "The minority class must contain at least n_splits observations "
            f"(found {minority_count}, requested {n_splits})."
        )

    cv = get_repeated_cv_strategy(n_splits, n_repeats, random_state)
    total_splits = n_splits * n_repeats
    rows: list[dict[str, Any]] = []
    candidate_names: list[str] = []

    for feature_config in selected_configs:
        for model_name, estimator in models.items():
            candidate_name = f"{model_name} | {feature_config}"
            candidate_names.append(candidate_name)
            scores = cross_validate(
                build_model_pipeline(estimator, feature_config=feature_config),
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
                    "candidate": candidate_name,
                    "model": model_name,
                    "feature_config": feature_config,
                    "repeat": split_index // n_splits + 1,
                    "fold": split_index % n_splits + 1,
                    "fit_time": float(scores["fit_time"][split_index]),
                    "score_time": float(scores["score_time"][split_index]),
                }
                for metric_name in METRIC_NAMES:
                    row[metric_name] = float(
                        scores[f"test_{metric_name}"][split_index]
                    )
                rows.append(row)

    fold_scores = pd.DataFrame(rows)
    metric_columns = list(METRIC_NAMES) + ["fit_time", "score_time"]
    grouped = fold_scores.groupby(
        ["candidate", "model", "feature_config"], sort=False
    )[metric_columns].agg(["mean", "std", "median", "min", "max"])
    grouped.columns = [
        f"{metric}_{statistic}" for metric, statistic in grouped.columns
    ]
    summary = grouped.reset_index()
    summary.insert(3, "fold_count", total_splits)

    repeat_means = (
        fold_scores.groupby(["repeat", "candidate"], as_index=False)["roc_auc"]
        .mean()
        .rename(columns={"roc_auc": "mean_roc_auc"})
    )
    winner_indices = repeat_means.groupby("repeat")["mean_roc_auc"].idxmax()
    repeat_winners = (
        repeat_means.loc[winner_indices]
        .sort_values("repeat")
        .rename(columns={"candidate": "selected_candidate"})
        .reset_index(drop=True)
    )
    wins = repeat_winners["selected_candidate"].value_counts()
    selection_stability = pd.DataFrame(
        {
            "candidate": candidate_names,
            "winning_repeats": [
                int(wins.get(name, 0)) for name in candidate_names
            ],
        }
    )
    selection_stability["selection_frequency"] = (
        selection_stability["winning_repeats"] / n_repeats
    )
    selection_stability = selection_stability.sort_values(
        ["winning_repeats", "candidate"], ascending=[False, True]
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
            "models": list(models),
            "feature_configs": selected_configs,
            "candidate_count": len(candidate_names),
            "test_set_used": False,
        },
    }

PHASE_2C_CANDIDATES = (
    "XGBoost (unweighted)",
    "XGBoost (balanced)",
    "Random Forest (balanced)",
)


def evaluate_phase_2c_nested_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_iter: int = 12,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Tune the three Phase 2A finalists and evaluate selection with nested CV.

    Each outer validation fold is kept entirely outside inner hyperparameter
    search and cross-family selection. Every candidate receives the same inner
    folds and the same number of randomized trials. The function accepts only
    the training partition; it has no interface for held-out test data.

    ROC-AUC is the predeclared selection metric. All other metrics are reported
    at the estimator's default 0.50 classification threshold and are secondary.
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    from .train import get_imbalance_experiment_models, get_tuning_distributions

    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("X_train must be a DataFrame and y_train must be a Series.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if y_train.isna().any() or set(y_train.unique()) != {0, 1}:
        raise ValueError("y_train must contain both binary labels 0 and 1.")
    if outer_splits < 2:
        raise ValueError("outer_splits must be at least 2.")
    if inner_splits < 2:
        raise ValueError("inner_splits must be at least 2.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    if int(y_train.value_counts().min()) < outer_splits:
        raise ValueError("The minority class must contain at least outer_splits observations.")

    all_candidates = get_imbalance_experiment_models()
    candidates = {name: all_candidates[name] for name in PHASE_2C_CANDIDATES}
    distributions = get_tuning_distributions()
    distribution_key = {
        "XGBoost (unweighted)": "XGBoost",
        "XGBoost (balanced)": "XGBoost",
        "Random Forest (balanced)": "Random Forest",
    }

    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []

    for outer_fold, (train_indices, valid_indices) in enumerate(
        outer_cv.split(X_train, y_train), start=1
    ):
        X_outer_train = X_train.iloc[train_indices]
        y_outer_train = y_train.iloc[train_indices]
        X_outer_valid = X_train.iloc[valid_indices]
        y_outer_valid = y_train.iloc[valid_indices]

        if int(y_outer_train.value_counts().min()) < inner_splits:
            raise ValueError(
                "Each outer training fold must contain at least inner_splits "
                "examples from the minority class."
            )

        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=random_state + outer_fold,
        )
        searches: dict[str, Any] = {}
        for candidate_name, estimator in candidates.items():
            search = RandomizedSearchCV(
                estimator=build_model_pipeline(estimator, feature_config="original"),
                param_distributions=distributions[distribution_key[candidate_name]],
                n_iter=n_iter,
                scoring=SCORING,
                refit="roc_auc",
                cv=inner_cv,
                random_state=random_state + outer_fold,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_outer_train, y_outer_train)
            searches[candidate_name] = search
            best_index = int(search.best_index_)
            inner_rows.append(
                {
                    "outer_fold": outer_fold,
                    "candidate": candidate_name,
                    "inner_best_roc_auc_mean": float(search.best_score_),
                    "inner_best_roc_auc_std": float(
                        search.cv_results_["std_test_roc_auc"][best_index]
                    ),
                    "inner_best_average_precision_mean": float(
                        search.cv_results_["mean_test_average_precision"][best_index]
                    ),
                    "inner_best_f1_mean": float(
                        search.cv_results_["mean_test_f1"][best_index]
                    ),
                    "best_params": dict(search.best_params_),
                    "trials_evaluated": n_iter,
                }
            )

        selected_name = max(
            candidates, key=lambda name: searches[name].best_score_
        )
        selected_search = searches[selected_name]
        selected_pipeline = selected_search.best_estimator_
        classifier = selected_pipeline.named_steps["classifier"]
        positive_class_index = list(classifier.classes_).index(1)
        probabilities = selected_pipeline.predict_proba(X_outer_valid)[
            :, positive_class_index
        ]
        predictions = selected_pipeline.predict(X_outer_valid)

        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_candidate": selected_name,
                "inner_best_roc_auc": float(selected_search.best_score_),
                "outer_roc_auc": float(roc_auc_score(y_outer_valid, probabilities)),
                "outer_average_precision": float(
                    average_precision_score(y_outer_valid, probabilities)
                ),
                "outer_precision": float(
                    precision_score(y_outer_valid, predictions, zero_division=0)
                ),
                "outer_recall": float(
                    recall_score(y_outer_valid, predictions, zero_division=0)
                ),
                "outer_f1": float(
                    f1_score(y_outer_valid, predictions, zero_division=0)
                ),
                "outer_accuracy": float(
                    accuracy_score(y_outer_valid, predictions)
                ),
                "outer_train_rows": len(train_indices),
                "outer_validation_rows": len(valid_indices),
                "best_params": dict(selected_search.best_params_),
            }
        )

    outer_results = pd.DataFrame(outer_rows)
    inner_results = pd.DataFrame(inner_rows)
    metric_columns = [
        "outer_roc_auc",
        "outer_average_precision",
        "outer_precision",
        "outer_recall",
        "outer_f1",
        "outer_accuracy",
    ]
    outer_summary = pd.DataFrame(
        [
            {
                "metric": metric.removeprefix("outer_"),
                "mean": float(outer_results[metric].mean()),
                "std": float(outer_results[metric].std(ddof=1)),
                "min": float(outer_results[metric].min()),
                "max": float(outer_results[metric].max()),
            }
            for metric in metric_columns
        ]
    )
    selected_counts = outer_results["selected_candidate"].value_counts()
    selection_frequency = pd.DataFrame(
        {
            "candidate": list(candidates),
            "selected_outer_folds": [
                int(selected_counts.get(name, 0)) for name in candidates
            ],
        }
    )
    selection_frequency["selection_frequency"] = (
        selection_frequency["selected_outer_folds"] / outer_splits
    )
    selection_frequency = selection_frequency.sort_values(
        ["selected_outer_folds", "candidate"], ascending=[False, True]
    ).reset_index(drop=True)

    return {
        "outer_fold_results": outer_results,
        "inner_candidate_results": inner_results,
        "outer_summary": outer_summary,
        "selection_frequency": selection_frequency,
        "configuration": {
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "n_iter_per_candidate_per_outer_fold": n_iter,
            "candidate_names": list(candidates),
            "feature_config": "original",
            "selection_metric": "roc_auc",
            "classification_threshold": 0.5,
            "random_state": random_state,
            "training_rows": len(X_train),
            "test_set_used": False,
            "selection_procedure_evaluated": True,
        },
    }

