# Project Status — Customer Churn Prediction

## Current state

**Batch 0 — Repository Stabilization: in progress**

The repository is being stabilized on the branch `fix/batch-0-stabilization`. The `main` branch remains unchanged until the pull request is reviewed and merged.

## Project objective

Build a reproducible machine-learning workflow that predicts customer churn using the IBM Telco Customer Churn dataset.

## Dataset

- Source: IBM Telco Customer Churn dataset (Kaggle)
- Expected file: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- The dataset is intentionally not committed to Git.
- Target: `Churn` (Yes/No)
- The notebook and training code must report observed dataset statistics at runtime rather than assume them.

## Batch 0 scope

- [x] Create a dedicated stabilization branch.
- [x] Remove confirmed temporary scripts and duplicate root-level EDA source.
- [x] Make the EDA notebook Colab-compatible and remove saved execution outputs.
- [x] Document Colab and local setup.
- [x] Define a supported Python version range.
- [x] Keep deterministic cleaning separate from learned imputation.
- [x] Move imputation, scaling, and encoding into reusable scikit-learn pipelines.
- [x] Perform cross-validation on raw training features through the full pipeline.
- [x] Select candidate models using cross-validation metrics only.
- [x] Reserve the held-out test set for final evaluation of the selected model.
- [x] Derive class weights from the current training labels.
- [x] Add pytest unit tests and GitHub Actions workflow.
- [ ] Confirm CI passes on the stabilization branch.
- [ ] Review the final diff and open a pull request.

## Leakage controls

1. Split the rows before fitting preprocessing.
2. Fit imputers, encoders, and scalers within each cross-validation fold through a complete sklearn Pipeline.
3. Select the candidate model using cross-validation results from the training split.
4. Evaluate only the selected model on the held-out test split.
5. Never use precomputed transformed arrays for cross-validation.

## Runtime and reproducibility

- Supported Python: 3.11–3.12
- CI Python: 3.12
- Random seed: 42
- Tests: `python -m pytest`

## Next action

Complete branch validation, inspect the final diff, and open a reviewable pull request. Do not merge without explicit approval.
