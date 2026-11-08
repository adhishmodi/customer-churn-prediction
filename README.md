# Customer Churn Prediction

An end-to-end machine-learning project that predicts customer churn using the IBM Telco Customer Churn dataset.

## Objective

Predict whether a telecom customer is likely to leave based on demographics, subscribed services, contract details, tenure, and billing information.

## Repository structure

- `notebooks/`: academic analysis and experiments (Google Colab-compatible)
- `src/`: reusable data loading, validation, preprocessing, and training code
- `tests/`: automated unit and regression tests
- `data/raw/`: original dataset (not committed)
- `data/processed/`: generated intermediate data (not committed)
- `models/`: generated model artifacts (not committed)
- `reports/`: generated figures and reports (not committed)

## Google Colab

1. Open the [EDA notebook in Google Colab](https://colab.research.google.com/github/adhishmodi/customer-churn-prediction/blob/main/notebooks/01_eda.ipynb).
2. Run the setup cell. It clones this repository into the Colab runtime if needed.
3. When prompted, upload the original CSV named `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
4. Run the notebook from top to bottom.

The dataset is not stored in this repository. Download the IBM Telco Customer Churn CSV from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) before uploading it. Keep the raw CSV unchanged.

## Local setup (Python 3.12 recommended)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the dataset at:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Run automated tests:

```bash
python -m pytest
```

## Project workflow

1. Load and validate the source dataset.
2. Explore the data and document evidence-based findings.
3. Split the data before fitting preprocessing transformations.
4. Train candidate models using cross-validation on the training split.
5. Select a candidate using cross-validation results.
6. Evaluate the selected model on the held-out test set once.

## Reproducibility and data leakage

The project uses a fixed random seed (42). Imputation, scaling, and categorical encoding belong inside a scikit-learn pipeline so that each cross-validation fold learns transformations only from its own training fold. The test split is reserved for final evaluation and is not used for model selection.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the current implementation status and [AI_DEVELOPMENT_RULES.md](AI_DEVELOPMENT_RULES.md) for project standards.
