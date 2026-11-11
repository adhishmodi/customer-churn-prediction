# Customer Churn Prediction

An end-to-end machine-learning project that predicts whether a telecom customer is likely to leave using the IBM Telco Customer Churn dataset.

## Final project result

The final Milestone 5 candidate is **balanced XGBoost with sigmoid probability calibration**. Its decision threshold (**0.344207**) was selected using training-only out-of-fold predictions.

On the project's 1,409-row final comparison set, this configuration achieved:

| Metric | Final model |
|---|---:|
| Accuracy | 76.58% |
| Precision | 54.38% |
| Recall | 72.99% |
| F1-score | 62.33% |
| ROC-AUC | 84.55% |
| Average Precision | 66.30% |

The final model improves accuracy and precision over the historical Milestone 4 threshold-0.53 reference, but has slightly lower recall, F1, ROC-AUC, and Average Precision. It is therefore **not an across-the-board improvement**. The same holdout was used in the earlier Milestone 4 evaluation, so this is a final historical comparison rather than an independent external validation.

See [the academic final report](docs/FINAL_REPORT.md) for the full methodology, experiment results, limitations, and interpretation. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the milestone record.

## Project objective

Predict customer churn from customer demographics, subscribed services, contract details, tenure, and billing information. The project demonstrates a reproducible ML workflow from exploratory analysis through model evaluation.

## Dataset

- Source: [IBM Telco Customer Churn dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Observations: 7,043 customers
- Original columns: 21 (customer identifier, 19 predictors, and churn target)
- Target distribution: 5,174 non-churn customers (73.46%) and 1,869 churn customers (26.54%)
- The raw CSV is not committed to this repository.

Expected filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Academic workflow and notebooks

All notebooks are Google Colab-compatible and are stored in `notebooks/`.

| Notebook | Purpose |
|---|---|
| [01 — EDA](notebooks/01_eda.ipynb) | Data quality, target distribution, feature distributions, and exploratory findings |
| [02 — Preprocessing](notebooks/02_preprocessing.ipynb) | Deterministic cleaning, stratified split, and training-fitted preprocessing |
| [03 — Model training](notebooks/03_model_training.ipynb) | Initial model comparison and randomized hyperparameter search |
| [04 — Model evaluation](notebooks/04_model_evaluation.ipynb) | Historical baseline evaluation and training-only threshold selection |
| [05 — Model improvement](notebooks/05_model_improvement.ipynb) | Repeated and nested cross-validation framework |
| [06 — Class imbalance](notebooks/06_class_imbalance_experiments.ipynb) | Balanced versus unweighted model comparisons |
| [07 — Calibration and thresholds](notebooks/07_calibration_threshold_optimization.ipynb) | Nested calibration and F1 threshold experiments |
| [08 — Final evaluation](notebooks/08_final_model_selection_and_evaluation.ipynb) | Finalist freeze and final historical holdout comparison |
| [09 — Feature engineering](notebooks/09_feature_engineering.ipynb) | Training-only comparison of tenure groups and subscribed-service count |
| [10 — Hyperparameter optimization](notebooks/10_hyperparameter_optimization.ipynb) | Nested training-only randomized search for the three retained finalists |

Open notebooks directly in Colab using the links below:

- [Open EDA notebook in Colab](https://colab.research.google.com/github/adhishmodi/customer-churn-prediction/blob/main/notebooks/01_eda.ipynb)
- [Open final evaluation notebook in Colab](https://colab.research.google.com/github/adhishmodi/customer-churn-prediction/blob/main/notebooks/08_final_model_selection_and_evaluation.ipynb)

## Methodology

1. Preserve the raw CSV and perform deterministic cleaning on a separate copy.
2. Remove `customerID` and encode `Churn` as a binary target.
3. Create a stratified 80/20 split using random seed 42 (5,634 training rows and 1,409 holdout rows).
4. Fit preprocessing inside complete scikit-learn pipelines: median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
5. Compare candidate classifiers using stratified cross-validation. Hyperparameter search uses training data only.
6. Compare class-balanced and unweighted candidates with repeated cross-validation.
7. Compare probability calibration methods and select an F1 threshold using nested training-only validation.
8. Freeze the finalist and threshold before calculating the final holdout metrics.

The original holdout was previously evaluated in Milestone 4. It must not be used for further tuning, feature selection, calibration, threshold selection, or model selection.

## Local setup

Python 3.12 is recommended (supported range: Python 3.11–3.12).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the original CSV at:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Run automated tests:

```powershell
python -m pytest
```

Optional dependencies for explainability, Streamlit, and local notebook tooling are listed in `requirements-optional.txt`.

## Repository structure

- `notebooks/`: Colab-compatible academic workflow and experiment notebooks
- `src/`: reusable data loading, validation, preprocessing, training, calibration, and evaluation code
- `tests/`: unit and regression tests
- `data/raw/`, `data/processed/`: local data (not committed)
- `models/`, `reports/`: generated artifacts (ignored by Git)
- `docs/FINAL_REPORT.md`: academic project report

## Reproducibility and data-leakage safeguards

- Random seed: 42.
- Use stratified splits for classification.
- Keep learned preprocessing inside the estimator pipeline so each CV fold fits transformations only on its own training rows.
- Use training-only out-of-fold predictions for calibration and threshold selection.
- Do not tune against the final holdout. Any future validation must use training-only resampling or a genuinely new external dataset.

## Project status

Milestones 1–4 are complete. Milestone 5 phases 0–1, 2A, 2B, 3, 4, and 5 are merged. Phase 2B's corrected Colab results were reviewed; original features were retained because engineered variants did not show a consistent validation improvement. Phase 2C implementation is on a feature branch; its Colab run and result review are pending.

For detailed results and limitations, read [the final report](docs/FINAL_REPORT.md) and [the project status](PROJECT_STATUS.md).
