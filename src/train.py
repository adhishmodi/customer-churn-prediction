"""Leakage-safe model training and evaluation for customer churn prediction."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CV_FOLDS, MODELS_DIR, RANDOM_SEED
from .preprocess import (
    clean_data,
    get_preprocessing_pipeline,
    load_and_clean_data,
    prepare_features_target,
    split_data,
)

logger = logging.getLogger(__name__)


def calculate_class_weights(y_train: pd.Series) -> tuple[dict[int, float], float]:
    """Derive balanced class weights and XGBoost positive-class ratio from train labels."""
    counts = y_train.value_counts()
    if set(counts.index.tolist()) != {0, 1}:
        raise ValueError("Training labels must contain both classes 0 and 1.")
    total = len(y_train)
    class_weights = {
        0: total / (2 * int(counts.loc[0])),
        1: total / (2 * int(counts.loc[1])),
    }
    return class_weights, float(counts.loc[0] / counts.loc[1])


class FoldBalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost wrapper that derives imbalance weights from each fit's labels.

    This keeps class-ratio information fold-local during cross-validation.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = RANDOM_SEED,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any) -> "FoldBalancedXGBClassifier":
        labels = np.asarray(y, dtype=int)
        negatives = int(np.sum(labels == 0))
        positives = int(np.sum(labels == 1))
        if not negatives or not positives:
            raise ValueError("XGBoost training requires both churn classes.")
        self.scale_pos_weight_ = negatives / positives
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


def get_models() -> dict[str, Any]:
    """Return candidates whose imbalance handling is fitted within each CV fold."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=1500,
            solver="lbfgs",
            C=1.0,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight=class_weights,
            random_state=RANDOM_SEED,
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=1,
        ),
        "XGBoost": FoldBalancedXGBClassifier(),
        "Support Vector Machine": SVC(
            class_weight=class_weights,
            random_state=RANDOM_SEED,
            probability=True,
            C=1.0,
            kernel="rbf",
            gamma="scale",
        ),
    }


def build_model_pipeline(estimator: Any) -> Pipeline:
    """Place all learned preprocessing inside the estimator's CV pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", get_preprocessing_pipeline()),
            ("classifier", estimator),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate one already-selected model on the untouched test split."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
    }


def cross_validate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    """Cross-validate the complete pipeline using training rows only."""
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=1,
        error_score="raise",
    )
    metrics: dict[str, float] = {}
    for metric in ("roc_auc", "precision", "recall", "f1"):
        values = scores[f"test_{metric}"]
        metrics[f"cv_{metric}_mean"] = float(values.mean())
        metrics[f"cv_{metric}_std"] = float(values.std())
    return metrics


def get_best_model(
    results: dict[str, dict[str, Any]],
    metric: str = "roc_auc",
) -> tuple[str, Pipeline]:
    """Select the candidate using cross-validation, never held-out test metrics."""
    if not results:
        raise ValueError("No model results were supplied.")
    cv_key = metric if metric.startswith("cv_") else f"cv_{metric}_mean"
    if not cv_key.endswith("_mean"):
        cv_key = f"{cv_key}_mean"
    if any(cv_key not in result["cv_metrics"] for result in results.values()):
        raise KeyError(f"Cross-validation metric '{cv_key}' is unavailable.")
    best_name = max(
        results,
        key=lambda name: results[name]["cv_metrics"][cv_key],
    )
    return best_name, results[best_name]["model"]


def train_and_evaluate_all(
    dataframe: pd.DataFrame | None = None,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Cross-validate all candidates, then evaluate only the CV-selected model on test."""
    cleaned = load_and_clean_data() if dataframe is None else clean_data(dataframe)
    X, y = prepare_features_target(cleaned)
    X_train, X_test, y_train, y_test = split_data(X, y)

    estimators = get_models()
    results: dict[str, dict[str, Any]] = {}

    for name, estimator in estimators.items():
        pipeline = build_model_pipeline(estimator)
        logger.info("Cross-validating %s", name)
        cv_metrics = cross_validate_model(pipeline, X_train, y_train)

        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start_time
        results[name] = {
            "model": pipeline,
            "train_time": fit_seconds,
            "cv_metrics": cv_metrics,
        }

    best_name, best_model = get_best_model(results, "roc_auc")
    results[best_name]["test_metrics"] = evaluate_model(
        best_model, X_test, y_test
    )
    feature_names = (
        best_model.named_steps["preprocessor"].get_feature_names_out().tolist()
    )
    return results, X_train, X_test, y_train, y_test, feature_names


def save_models(
    results: dict[str, dict[str, Any]],
    output_dir: Path = MODELS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(result["model"], output_dir / filename)


def save_results_summary(
    results: dict[str, dict[str, Any]],
    output_dir: Path = MODELS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, result in results.items():
        row: dict[str, Any] = {"model": name, "train_time_seconds": result["train_time"]}
        row.update(result["cv_metrics"])
        if "test_metrics" in result:
            row.update({f"test_{key}": value for key, value in result["test_metrics"].items()})
            row["evaluated_on_test"] = True
        else:
            row["evaluated_on_test"] = False
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "model_comparison.csv", index=False)


def print_comparison_table(results: dict[str, dict[str, Any]]) -> None:
    """Print CV metrics for all candidates and test metrics for the selected model only."""
    rows = []
    for name, result in results.items():
        row = {"Model": name, **result["cv_metrics"]}
        if "test_metrics" in result:
            row.update({f"test_{key}": value for key, value in result["test_metrics"].items()})
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def run_training() -> dict[str, dict[str, Any]]:
    results, _, _, _, _, _ = train_and_evaluate_all()
    print_comparison_table(results)
    save_models(results)
    save_results_summary(results)
    best_name, _ = get_best_model(results, "roc_auc")
    logger.info("CV-selected model: %s", best_name)
    logger.info("Held-out test metrics: %s", results[best_name]["test_metrics"])
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
