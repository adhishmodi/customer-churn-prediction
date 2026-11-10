"""Leakage-safe model comparison and tuning for customer churn prediction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, make_scorer,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import CV_FOLDS, RANDOM_SEED
from .preprocess import get_preprocessing_pipeline

logger = logging.getLogger(__name__)

SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "accuracy": "accuracy",
}


class FoldBalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost wrapper with optional fold-local positive-class weighting."""

    def __init__(
        self, n_estimators: int = 120, max_depth: int = 4,
        learning_rate: float = 0.05, subsample: float = 0.8,
        colsample_bytree: float = 0.8, random_state: int = RANDOM_SEED,
        n_jobs: int = 1,
        balance_classes: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.balance_classes = balance_classes

    def fit(self, X: Any, y: Any) -> "FoldBalancedXGBClassifier":
        labels = np.asarray(y, dtype=int)
        negatives = int(np.sum(labels == 0))
        positives = int(np.sum(labels == 1))
        if not negatives or not positives:
            raise ValueError("XGBoost training requires both churn classes.")
        self.scale_pos_weight_ = (
            negatives / positives if self.balance_classes else 1.0
        )
        self.estimator_ = xgb.XGBClassifier(
            scale_pos_weight=self.scale_pos_weight_,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=self.n_jobs,
        )
        self.estimator_.fit(X, labels)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)


def calculate_class_weights(y_train: pd.Series) -> tuple[dict[int, float], float]:
    """Calculate balanced class weights using only the supplied training labels."""
    counts = y_train.value_counts()
    if set(counts.index.tolist()) != {0, 1}:
        raise ValueError("Training labels must contain both classes 0 and 1.")
    total = len(y_train)
    weights = {
        0: total / (2 * int(counts.loc[0])),
        1: total / (2 * int(counts.loc[1])),
    }
    return weights, float(counts.loc[0] / counts.loc[1])


def get_models() -> dict[str, BaseEstimator]:
    """Create baseline and candidate classifiers with reproducible settings."""
    return {
        "Dummy": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=RANDOM_SEED,
            max_iter=1500, solver="lbfgs", C=1.0,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_SEED,
            max_depth=10, min_samples_split=20, min_samples_leaf=10,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_SEED,
            n_estimators=150, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, n_jobs=1,
        ),
        "XGBoost": FoldBalancedXGBClassifier(),
    }


def get_imbalance_experiment_models() -> dict[str, BaseEstimator]:
    """Create matched weighted/unweighted candidates for Phase 2A.

    All non-weighting hyperparameters are copied from the existing initial
    model configurations. The Dummy prior classifier is a prevalence-only
    reference, not a weighted/unweighted pair.
    """
    initial_models = get_models()
    candidates: dict[str, BaseEstimator] = {
        "Dummy (prior)": clone(initial_models["Dummy"]),
    }
    for model_name in (
        "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"
    ):
        balanced = clone(initial_models[model_name])
        unweighted = clone(initial_models[model_name])
        if model_name == "XGBoost":
            unweighted.set_params(balance_classes=False)
        else:
            unweighted.set_params(class_weight=None)
        candidates[f"{model_name} (balanced)"] = balanced
        candidates[f"{model_name} (unweighted)"] = unweighted
    return candidates


def build_model_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Keep every learned preprocessing step inside the estimator pipeline."""
    return Pipeline([
        ("preprocessor", get_preprocessing_pipeline()),
        ("classifier", estimator),
    ])


def get_cv_strategy() -> StratifiedKFold:
    """Return the project's reproducible stratified cross-validation strategy."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


def _summarize_cv_scores(scores: dict[str, np.ndarray]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        metrics[f"cv_{metric}_mean"] = float(np.mean(values))
        metrics[f"cv_{metric}_std"] = float(np.std(values))
    return metrics


def cross_validate_model(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
) -> dict[str, float]:
    """Cross-validate a complete preprocessing/model pipeline on training rows."""
    scores = cross_validate(
        model, X_train, y_train, cv=get_cv_strategy(), scoring=SCORING,
        n_jobs=1, error_score="raise", return_train_score=False,
    )
    metrics = _summarize_cv_scores(scores)
    metrics["cv_fit_time_mean"] = float(np.mean(scores["fit_time"]))
    return metrics


def compare_models(
    X_train: pd.DataFrame, y_train: pd.Series,
    models: dict[str, BaseEstimator] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare initial candidates using CV only; test data is not accepted."""
    candidates = get_models() if models is None else models
    results: dict[str, dict[str, Any]] = {}
    for name, estimator in candidates.items():
        logger.info("Cross-validating %s", name)
        pipeline = build_model_pipeline(estimator)
        results[name] = {
            "model": pipeline,
            "cv_metrics": cross_validate_model(pipeline, X_train, y_train),
            "model_type": "baseline" if name == "Dummy" else "initial",
        }
    return results


def get_tuning_distributions() -> dict[str, dict[str, list[Any]]]:
    """Return compact, reproducible search spaces for the three tunable models."""
    return {
        "Logistic Regression": {
            "classifier__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 150, 200, 250],
            "classifier__max_depth": [None, 8, 12, 16],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", 0.8],
        },
        "XGBoost": {
            "classifier__n_estimators": [80, 120, 160, 200],
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.7, 0.85, 1.0],
        },
    }


def tune_models(
    X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 10,
    model_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Tune selected models with randomized stratified CV on training data only."""
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    base_models = get_models()
    spaces = get_tuning_distributions()
    selected_names = list(spaces) if model_names is None else model_names
    unknown = sorted(set(selected_names) - set(spaces))
    if unknown:
        raise ValueError(f"Models do not have tuning spaces: {unknown}")

    results: dict[str, dict[str, Any]] = {}
    for name in selected_names:
        search = RandomizedSearchCV(
            estimator=build_model_pipeline(base_models[name]),
            param_distributions=spaces[name],
            n_iter=n_iter,
            scoring=SCORING,
            refit="roc_auc",
            cv=get_cv_strategy(),
            random_state=RANDOM_SEED,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        logger.info("Tuning %s with %d randomized candidates", name, n_iter)
        search.fit(X_train, y_train)
        best_index = int(search.best_index_)
        metrics: dict[str, float] = {}
        for metric in SCORING:
            metrics[f"cv_{metric}_mean"] = float(
                search.cv_results_[f"mean_test_{metric}"][best_index]
            )
            metrics[f"cv_{metric}_std"] = float(
                search.cv_results_[f"std_test_{metric}"][best_index]
            )
        results[f"Tuned {name}"] = {
            "model": search.best_estimator_,
            "cv_metrics": metrics,
            "best_params": dict(search.best_params_),
            "model_type": "tuned",
        }
    return results


def get_best_model(
    results: dict[str, dict[str, Any]], metric: str = "roc_auc",
) -> tuple[str, Pipeline]:
    """Select the highest mean CV score; test metrics are never considered."""
    if not results:
        raise ValueError("No model results were supplied.")
    cv_key = metric if metric.startswith("cv_") else f"cv_{metric}_mean"
    if not cv_key.endswith("_mean"):
        cv_key = f"{cv_key}_mean"
    if any(cv_key not in result["cv_metrics"] for result in results.values()):
        raise KeyError(f"Cross-validation metric '{cv_key}' is unavailable.")
    best_name = max(results, key=lambda name: results[name]["cv_metrics"][cv_key])
    return best_name, results[best_name]["model"]


def build_comparison_dataframe(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Convert CV results to a sortable report without held-out test metrics."""
    rows = [
        {"model": name, "model_type": result.get("model_type", "candidate"),
         **result["cv_metrics"]}
        for name, result in results.items()
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty and "cv_roc_auc_mean" in frame:
        frame = frame.sort_values("cv_roc_auc_mean", ascending=False)
    return frame.reset_index(drop=True)


def save_candidate_model(model: Pipeline, path: Path) -> None:
    """Persist the selected fitted pipeline, including its fitted preprocessor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_comparison_report(
    results: dict[str, dict[str, Any]], path: Path,
) -> pd.DataFrame:
    """Write the model comparison report using CV metrics only."""
    report = build_comparison_dataframe(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return report


def save_training_metadata(
    model_name: str, cv_metrics: dict[str, float],
    best_params: dict[str, Any] | None, path: Path,
) -> None:
    """Save reproducibility and selection metadata without storing source data."""
    import json

    payload = {
        "selected_model": model_name,
        "selection_metric": "cv_roc_auc_mean",
        "cv_folds": CV_FOLDS,
        "random_seed": RANDOM_SEED,
        "cv_metrics": cv_metrics,
        "best_params": best_params or {},
        "test_set_evaluated": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a frozen model; this helper is reserved for Milestone 4."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
    }
