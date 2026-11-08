\"\"\"Model training and evaluation for Customer Churn Prediction.\"\"\"
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb

from .config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, CV_FOLDS


def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    X_train = np.load(PROCESSED_DATA_DIR / \"X_train.npy\")
    X_test = np.load(PROCESSED_DATA_DIR / \"X_test.npy\")
    y_train = np.load(PROCESSED_DATA_DIR / \"y_train.npy\")
    y_test = np.load(PROCESSED_DATA_DIR / \"y_test.npy\")
    with open(PROCESSED_DATA_DIR / \"feature_names.txt\", \"r\") as f:
        feature_names = [line.strip() for line in f]
    return X_train, X_test, y_train, y_test, feature_names


def load_class_weights() -> Tuple[dict, float]:
    class_weights = {0: 0.6806, 1: 1.8843}
    scale_pos_weight = 2.77
    return class_weights, scale_pos_weight


def get_models(class_weights: dict, scale_pos_weight: float) -> Dict[str, Any]:
    models = {
        \"Logistic Regression\": LogisticRegression(
            class_weight=class_weights,
            random_state=RANDOM_SEED,
            max_iter=1000,
            solver=\"lbfgs\",
            C=1.0
        ),
        \"Decision Tree\": DecisionTreeClassifier(
            class_weight=class_weights,
            random_state=RANDOM_SEED,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10
        ),
        \"Random Forest\": RandomForestClassifier(
            class_weight=class_weights,
            random_state=RANDOM_SEED,
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1
        ),
        \"XGBoost\": xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric=\"logloss\",
            n_jobs=-1
        ),
        \"Support Vector Machine\": SVC(
            class_weight=class_weights,
            random_state=RANDOM_SEED,
            probability=True,
            C=1.0,
            kernel=\"rbf\",
            gamma=\"scale\"
        )
    }
    return models


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        \"roc_auc\": roc_auc_score(y_test, y_pred_proba),
        \"precision\": precision_score(y_test, y_pred, zero_division=0),
        \"recall\": recall_score(y_test, y_pred, zero_division=0),
        \"f1\": f1_score(y_test, y_pred, zero_division=0),
        \"accuracy\": accuracy_score(y_test, y_pred)
    }
    return metrics


def cross_validate_model(model, X_train: np.ndarray, y_train: np.ndarray, model_name: str) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring=\"roc_auc\", n_jobs=-1)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring=\"f1\", n_jobs=-1)
    cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring=\"precision\", n_jobs=-1)
    cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring=\"recall\", n_jobs=-1)
    return {
        \"cv_roc_auc_mean\": cv_roc_auc.mean(),
        \"cv_roc_auc_std\": cv_roc_auc.std(),
        \"cv_f1_mean\": cv_f1.mean(),
        \"cv_f1_std\": cv_f1.std(),
        \"cv_precision_mean\": cv_precision.mean(),
        \"cv_precision_std\": cv_precision.std(),
        \"cv_recall_mean\": cv_recall.mean(),
        \"cv_recall_std\": cv_recall.std()
    }


def train_and_evaluate_all() -> Tuple[Dict[str, Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    print(\"=\" * 60)
    print(\"STARTING MODEL TRAINING (BATCH 4)\")
    print(\"=\" * 60)
    print(\"\\n1. Loading processed data...\")
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()
    print(\"   Train: \" + str(X_train.shape) + \", Test: \" + str(X_test.shape))
    print(\"   Features: \" + str(len(feature_names)))
    print(\"\\n2. Loading class weights...\")
    class_weights, scale_pos_weight = load_class_weights()
    print(\"   Class weights: \" + str(class_weights))
    print(\"   scale_pos_weight: \" + str(scale_pos_weight))
    print(\"\\n3. Initializing models...\")
    models = get_models(class_weights, scale_pos_weight)
    for name in models:
        print(\"   - \" + name)
    print(\"\\n4. Training and evaluating models...\")
    results = {}
    for name, model in models.items():
        print(\"\\n   Training \" + name + \"...\")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        test_metrics = evaluate_model(model, X_test, y_test, name)
        print(\"   Cross-validating \" + name + \"...\")
        cv_metrics = cross_validate_model(model, X_train, y_train, name)
        results[name] = {
            \"model\": model,
            \"train_time\": train_time,
            \"test_metrics\": test_metrics,
            \"cv_metrics\": cv_metrics
        }
        print(\"   Train time: \" + str(round(train_time, 2)) + \"s\")
        print(\"   Test ROC-AUC: \" + str(round(test_metrics[\"roc_auc\"], 4)))
        print(\"   Test F1: \" + str(round(test_metrics[\"f1\"], 4)))
        cv_mean = round(cv_metrics[\"cv_roc_auc_mean\"], 4)
        cv_std = round(cv_metrics[\"cv_roc_auc_std\"], 4)
        print(\"   CV ROC-AUC: \" + str(cv_mean) + \" (+/- \" + str(cv_std) + \")\")
    return results, X_train, X_test, y_train, y_test, feature_names


def save_models(results: Dict[str, Dict], output_dir: Path = MODELS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        model_path = output_dir / (name.lower().replace(\" \", \"_\") + \".joblib\")
        joblib.dump(result[\"model\"], model_path)
        print(\"  Saved: \" + str(model_path))


def print_comparison_table(results: Dict[str, Dict]) -> None:
    print(\"\\n\" + \"=\" * 100)
    print(\"MODEL COMPARISON RESULTS\")
    print(\"=\" * 100)
    print(\"\\n--- Test Set Metrics ---\")
    print(\"{:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}\".format(\"Model\", \"ROC-AUC\", \"Precision\", \"Recall\", \"F1\", \"Accuracy\", \"Time(s)\"))
    print(\"-\" * 85)
    for name, result in results.items():
        m = result[\"test_metrics\"]
        t = result[\"train_time\"]
        print(\"{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>8.2f}\".format(name, m[\"roc_auc\"], m[\"precision\"], m[\"recall\"], m[\"f1\"], m[\"accuracy\"], t))
    print(\"\\n--- Cross-Validation Metrics (5-fold) ---\")
    print(\"{:<25} {:>15} {:>15} {:>15} {:>15}\".format(\"Model\", \"CV ROC-AUC\", \"CV F1\", \"CV Precision\", \"CV Recall\"))
    print(\"-\" * 85)
    for name, result in results.items():
        cv = result[\"cv_metrics\"]
        print(\"{:<25} {:.4f}+/-{:.4f}   {:.4f}+/-{:.4f}   {:.4f}+/-{:.4f}   {:.4f}+/-{:.4f}\".format(name, cv[\"cv_roc_auc_mean\"], cv[\"cv_roc_auc_std\"], cv[\"cv_f1_mean\"], cv[\"cv_f1_std\"], cv[\"cv_precision_mean\"], cv[\"cv_precision_std\"], cv[\"cv_recall_mean\"], cv[\"cv_recall_std\"]))


def save_results_summary(results: Dict[str, Dict], output_dir: Path = MODELS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, result in results.items():
        row = {\"model\": name}
        row.update({\"test_\" + k: v for k, v in result[\"test_metrics\"].items()})
        row.update({\"cv_\" + k: v for k, v in result[\"cv_metrics\"].items()})
        row[\"train_time\"] = result[\"train_time\"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / \"model_comparison.csv\", index=False)
    print(\"\\n  Saved comparison table: \" + str(output_dir / \"model_comparison.csv\"))


def get_best_model(results: Dict[str, Dict], metric: str = \"roc_auc\") -> Tuple[str, Any]:
    best_name = max(results.keys(), key=lambda k: results[k][\"test_metrics\"][metric])
    return best_name, results[best_name][\"model\"]


def run_training() -> Dict[str, Dict]:
    results, X_train, X_test, y_train, y_test, feature_names = train_and_evaluate_all()
    print_comparison_table(results)
    print(\"\\n5. Saving models...\")
    save_models(results)
    print(\"\\n6. Saving results summary...\")
    save_results_summary(results)
    best_name, best_model = get_best_model(results, \"roc_auc\")
    print(\"\\nBest model (by ROC-AUC): \" + best_name)
    print(\"   Test ROC-AUC: \" + str(round(results[best_name][\"test_metrics\"][\"roc_auc\"], 4)))
    print(\"   Test F1: \" + str(round(results[best_name][\"test_metrics\"][\"f1\"], 4)))
    print(\"\\n\" + \"=\" * 60)
    print(\"MODEL TRAINING COMPLETE\")
    print(\"=\" * 60)
    return results


if __name__ == \"__main__\":
    results = run_training()
