# Customer Churn Prediction

An end-to-end machine learning project using the IBM Telco Customer Churn dataset.

## Objective
Predict whether a telecom customer is likely to churn using customer demographics, subscribed services, contract details, and billing information.

## Project structure
- `data/raw/`: original, unmodified dataset
- `data/processed/`: cleaned or transformed data (generated)
- `notebooks/`: exploratory analysis and experiments
- `src/`: reusable project code
- `models/`: trained model artifacts (generated; do not commit large files)
- `reports/`: figures and project findings

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
Download the IBM Telco Customer Churn CSV from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV in `data/raw/`. Keep the raw dataset unchanged.

## Workflow
1. Inspect the dataset and define the prediction target.
2. Perform exploratory data analysis (EDA).
3. Build preprocessing and baseline models.
4. Compare models using appropriate classification metrics.
5. Document results and limitations.
