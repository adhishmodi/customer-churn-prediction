# Project Status — Customer Churn Prediction

## Current state

**Milestone 1 — Exploratory data analysis: completed and merged.**

**Milestone 2 — Data preprocessing: completed and merged.**

**Milestone 3 — Model training and selection: completed and merged.**

**Milestone 4 — Model evaluation: completed and merged.**

**Milestone 5 — Model improvement and optimization: approved; Phase 0 and Phase 1 in progress.**

Current Milestone 5 work is developed on branch
feature/milestone-5-validation-framework. Do not commit directly to main.

## Project objective

Build a reproducible, academically defensible machine-learning workflow that
predicts customer churn using the IBM Telco Customer Churn dataset.

## Dataset and split

- Source: IBM Telco Customer Churn dataset.
- Expected file: WA_Fn-UseC_-Telco-Customer-Churn.csv.
- Dataset size in the completed Milestone 4 run: 7,043 records.
- Target: Churn (Yes / No).
- Features used: 19 after excluding customerID and the target.
- Stratified split: 80% training (5,634 rows), 20% held-out test (1,409 rows).
- Random seed: 42.
- The raw CSV is intentionally excluded from Git.

## Frozen Milestone 3–4 baseline

The selected model is **Tuned XGBoost**, evaluated using the original
Milestone 3 training process and the held-out Milestone 4 test split.

### Training and validation results

- Five-fold shuffled StratifiedKFold, seed 42.
- Model selection metric: mean CV ROC-AUC.
- Hyperparameter search: RandomizedSearchCV, 10 iterations per tuned model.
- Tuned XGBoost CV ROC-AUC: **0.8501** (fold standard deviation 0.0119).
- Tuned XGBoost CV Average Precision: **0.6694** (fold standard deviation 0.0210).

### Original held-out test results at threshold 0.50

| Metric | Baseline result |
|---|---:|
| Accuracy | 74.02% |
| Precision | 50.67% |
| Recall | 81.02% |
| F1-score | 62.35% |
| ROC-AUC | 84.63% |
| Average Precision | 66.60% |

Original confusion matrix:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 740 | 295 |
| Actual Yes | 71 | 303 |

The training-only out-of-fold threshold of 0.53 was also reported in
Milestone 4. Its held-out test results were accuracy 75.09%, precision 52.02%,
recall 79.14%, and F1-score 62.78%. ROC-AUC and Average Precision remain
84.63% and 66.60%, respectively, because threshold changes do not change
probability ranking.

**Baseline preservation rule:** These values are historical reference
results. Do not overwrite, recompute from test data, or relabel them as
Milestone 5 results.

## Baseline artifact availability

The GitHub repository intentionally ignores generated artifacts:
models/*, reports/*, and *.joblib.

The current main tree contains only models/.gitkeep; the latest successful
GitHub Actions run (42617867013) has no downloadable artifacts. The executed
Milestone 4 notebook documents that the original Colab run required manual
upload of:

- best_model_candidate.joblib
- training_metadata.json
- WA_Fn-UseC_-Telco-Customer-Churn.csv

The exact original model binary and metadata are therefore **not currently
available in the Git repository or the latest Actions run**. The recorded
baseline metrics remain preserved above and in the executed notebooks.
Before a direct artifact-to-artifact comparison, recover the original files
from the original Colab session if possible. If they cannot be recovered,
reproduce the baseline through the committed training notebook and clearly
label it as a regenerated baseline; do not claim it is the identical original
binary.

## Existing architecture

- src/config.py: project paths, seed, split and CV constants.
- src/data_loader.py: CSV loading and basic data access.
- src/validate.py: dataset validation.
- src/preprocess.py: deterministic cleaning, feature/target separation,
  stratified splitting, and fold-fitted preprocessing pipeline.
- src/train.py: model factories, fold-balanced XGBoost, cross-validation,
  randomized tuning, candidate selection, and artifact helpers.
- notebooks/01_eda.ipynb through 04_model_evaluation.ipynb: completed
  academic workflow.
- tests/: unit and regression tests.
- .github/workflows/tests.yml: pytest on Python 3.12.

All learned preprocessing must remain inside the estimator pipeline so that
each CV fold fits imputers, scaling and encoding only on its own training rows.

## Milestone 5 roadmap

### Phase 0 — Baseline and repository preparation

- [x] Re-audit the latest main branch after PR #8.
- [x] Verify repository tree, open PRs/issues, recent Actions and workflow.
- [x] Preserve the original baseline metrics and document artifact availability.
- [ ] Recover the exact original Colab model and metadata artifacts, or arrange
      an explicitly labelled baseline regeneration before direct comparison.

### Phase 1 — Validation protocol and experiment framework

- [x] Add a repeated stratified CV strategy with fixed random seed.
- [x] Add training-only repeated-CV comparison for fixed model configurations.
- [x] Preserve fold-level scores, descriptive summaries and per-repeat winners.
- [ ] Run the new notebook against the original CSV in Colab and review outputs.
- [x] Confirm GitHub Actions and all tests on the Milestone 5 PR (28 passed; run 42670466818).

Repeated-CV fold standard deviation is descriptive variability. Because
repeated folds overlap, it must not be presented as a formal confidence
interval or a statistical significance test. Model-winner frequency is a
descriptive stability measure, not proof of superiority.

### Remaining Milestone 5 phases

- Phase 2: controlled class-imbalance and feature-engineering experiments.
- Phase 3: probability calibration and threshold/business-cost analysis.
- Phase 4: training-only candidate selection, freeze the final candidate,
  then perform one final held-out test evaluation.
- Phase 5: academic documentation, figures, tests, PR review and merge.

## Reproducibility and execution

- Supported Python: 3.11–3.12.
- CI Python: 3.12.
- Random seed: 42.
- Automated tests: python -m pytest.
- Use Google Colab or VS Code with the original CSV supplied locally.
- Never use the held-out test split for tuning, feature selection, calibration,
  threshold selection, or experiment decisions.

## Current next action

Run notebooks/05_model_improvement.ipynb with the original CSV in Colab.
Review repeated-CV fold results, descriptive summaries and model selection
stability before deciding which model-improvement experiments are justified.
