# Project Status — Customer Churn Prediction

## Current state

**Milestone 1 — Repository stabilization and EDA: merged**

- Batch 0 stabilization is merged into `main`.
- EDA conclusions are merged into `notebooks/01_eda.ipynb`.
- The IBM Telco Customer Churn CSV remains outside Git and is uploaded to Google Colab at runtime.
- Milestone 2 — Data Preprocessing is the current workstream.

## Project objective

Build a reproducible machine-learning workflow that predicts customer churn using the IBM Telco Customer Churn dataset.

## Dataset

- Source: IBM Telco Customer Churn dataset (Kaggle).
- Expected file: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- Target: `Churn` (`Yes` / `No`).
- The raw dataset is intentionally not committed to Git.
- The notebook and training code must report observed dataset statistics at runtime rather than assume them.

## Completed milestones

### Milestone 1 — Repository stabilization and EDA

- [x] Stabilize repository structure and Colab-compatible EDA notebook.
- [x] Keep raw data unchanged and document data quality findings.
- [x] Record EDA conclusions and modeling implications in Section 8 of `01_eda.ipynb`.
- [x] Add automated tests and GitHub Actions workflow.
- [x] Merge PR #1 (repository stabilization).
- [x] Merge PR #2 (EDA conclusions).

## Milestone 2 — Data preprocessing

### Design and leakage controls

1. Apply only deterministic, row-level cleaning before the split.
2. Convert `TotalCharges` to numeric and set its blank value to 0 only for zero-tenure customers.
3. Exclude `customerID` from model features.
4. Encode `Churn` as `No = 0` and `Yes = 1`.
5. Create a reproducible stratified 80/20 train/test split using random seed 42.
6. Fit numeric median imputation and standard scaling on training data only.
7. Fit categorical most-frequent imputation and one-hot encoding on training data only.
8. Transform test data with the fitted training-only preprocessor.
9. Keep cross-validation leakage-safe by placing the preprocessor and estimator in one pipeline and cross-validating on raw training features.

### Milestone 2 deliverables

- [ ] Create and run `notebooks/02_preprocessing.ipynb` in Google Colab with the original CSV.
- [ ] Verify deterministic cleaning, target encoding, split reproducibility, and stratification.
- [ ] Verify missing-value imputation, scaling, one-hot encoding, and unseen-category handling.
- [ ] Verify the preprocessor is fitted only on training rows.
- [ ] Save generated preprocessing artifacts under `data/processed/` (ignored by Git).
- [ ] Pass the preprocessing unit tests and GitHub Actions workflow.
- [ ] Review and merge the Milestone 2 pull request.

## Runtime and reproducibility

- Supported Python: 3.11–3.12.
- CI Python: 3.12.
- Random seed: 42.
- Tests: `python -m pytest`.

## Next action

Run the Milestone 2 notebook in Google Colab using the original dataset, review the observed outputs, and then proceed to model training only after preprocessing is verified.
