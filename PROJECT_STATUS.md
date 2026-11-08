# Project Status — Customer Churn Prediction

## Current state

**Milestone 1 — Repository stabilization and EDA: merged**

**Milestone 2 — Data preprocessing: implementation merged**

- The Colab-compatible preprocessing notebook and leakage-focused unit tests are merged.
- The IBM Telco Customer Churn CSV remains outside Git and is uploaded to Colab at runtime.
- The preprocessing notebook has been run with the source CSV; its observed outputs are available in notebook history. Re-run it from a clean runtime when reproducing the project.
- **Milestone 3 — Model training and CV-based model selection: in progress (PR review pending).**

## Project objective

Build a reproducible machine-learning workflow that predicts customer churn using the IBM Telco Customer Churn dataset.

## Dataset

- Source: IBM Telco Customer Churn dataset (Kaggle).
- Expected file: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.
- Target: Churn (Yes / No).
- The raw dataset is intentionally not committed to Git.
- Report observed dataset statistics at runtime; do not hard-code assumed values.

## Completed milestones

### Milestone 1 — Repository stabilization and EDA

- [x] Stabilize repository structure and Colab-compatible EDA notebook.
- [x] Record EDA conclusions and modeling implications.
- [x] Add automated tests and GitHub Actions workflow.
- [x] Merge PR #1 and PR #2.

### Milestone 2 — Data preprocessing

- [x] Add the Colab-compatible preprocessing notebook.
- [x] Implement deterministic cleaning, target encoding, and stratified train/test splitting.
- [x] Keep learned imputation, scaling, and encoding fitted on training data only.
- [x] Add preprocessing tests for data integrity and leakage safeguards.
- [x] Merge PR #3.

### Milestone 3 — Model training and selection

#### Design and leakage controls

1. Use deterministic cleaning and the reproducible stratified 80/20 split (seed 42).
2. Use raw training features for all cross-validation and tuning.
3. Keep the complete preprocessing transformer inside each model pipeline.
4. Use five-fold shuffled StratifiedKFold with seed 42.
5. Compare a Dummy baseline, Logistic Regression, Decision Tree, Random Forest, and XGBoost.
6. Use mean CV ROC-AUC as the primary selection metric.
7. Also report Average Precision, Precision, Recall, F1, and Accuracy.
8. Tune Logistic Regression, Random Forest, and XGBoost using randomized search on training data only.
9. Fit and save the selected candidate pipeline, including its fitted preprocessor.
10. Do not evaluate the held-out test split in this milestone; reserve it for Milestone 4.

#### Deliverables

- notebooks/03_model_training.ipynb: Colab-compatible, clean-output model training notebook.
- src/train.py: reusable model factories, leakage-safe CV, randomized tuning, selection, and artifact helpers.
- tests/test_training.py: tests for candidate models, CV metrics, tuning, CV-only selection, artifact reload, fold-local XGBoost class ratio, and notebook cleanliness.
- Generated local artifacts (ignored by Git):
  - models/best_model_candidate.joblib
  - models/training_metadata.json
  - reports/initial_model_comparison.csv
  - reports/model_comparison.csv

#### Validation status

- [ ] Run the notebook in Colab using the original CSV and inspect actual model results.
- [ ] Run the full pytest suite and verify GitHub Actions on the Milestone 3 PR.
- [ ] Review and merge the Milestone 3 pull request.

## Runtime and reproducibility

- Supported Python: 3.11–3.12.
- CI Python: 3.12.
- Random seed: 42.
- Tests: python -m pytest.

## Next action

Review the Milestone 3 pull request and run notebooks/03_model_training.ipynb in Colab with the original dataset. After reviewing the CV results, proceed to Milestone 4 for one-time held-out test evaluation.
