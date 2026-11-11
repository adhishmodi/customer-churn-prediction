# Project Status — Customer Churn Prediction

## Current state

**Milestone 1 — Exploratory data analysis: completed and merged.**

**Milestone 2 — Data preprocessing: completed and merged.**

**Milestone 3 — Model training and selection: completed and merged.**

**Milestone 4 — Model evaluation: completed and merged.**

**Milestone 5 — Model improvement and optimization: Phase 4 complete; Phase 5 documentation in progress.** Phases 0–1, 2A, 3 and 4 are merged. Phase 2A, Phase 3 and Phase 4 Colab outputs have been reviewed or committed. Phases 2B and 2C remain deferred. The final Phase 4 candidate is balanced XGBoost + sigmoid calibration with a training-only OOF threshold of 0.344207.

Phase 2A implementation was merged in PR #10. Follow-up work must use feature branches and pull requests; do not commit directly to main.

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
- [x] Add nested CV with inner-fold tuning/model selection and outer-fold evaluation.
- [x] Preserve fold-level scores, descriptive summaries and per-repeat winners.
- [x] Run the new notebook against the original CSV in Colab and review outputs.
- [x] Confirm GitHub Actions for the current implementation (30 passed; run 42671021280). A later documentation-only status update triggered a fresh CI rerun.

Repeated-CV fold standard deviation is descriptive variability. Because
repeated folds overlap, it must not be presented as a formal confidence
interval or a statistical significance test. Model-winner frequency is a
descriptive stability measure, not proof of superiority. Nested CV evaluates
the model-and-hyperparameter selection procedure using inner folds and
reserves outer folds for evaluation; its results remain to be generated in
Colab with the original CSV.

### Remaining Milestone 5 phases

- [x] Phase 2A: controlled class-imbalance experiments (PR #10 merged; Colab run completed; five CSV outputs reviewed).
- [ ] Phase 2B: feature-engineering experiments (deferred by project decision).
- [ ] Phase 2C: hyperparameter optimization (deferred by project decision).
- [x] Phase 3: nested probability calibration and F1-threshold optimization implemented (PR #12 merged); Colab results reviewed.
- [ ] Phase 4: final training-only finalist selection, calibration and threshold freeze, then final evaluation against the historical holdout (implementation in PR #13; Colab execution pending).
- [ ] Phase 5: academic documentation, figures, final tests, PR review and merge.


### Phase 2A — Colab results reviewed

The executed notebook and five exported CSV files were reviewed. Across 15
training-only repeated-CV folds, XGBoost unweighted achieved mean ROC-AUC 0.8490
and Average Precision 0.6701. XGBoost balanced achieved mean ROC-AUC 0.8484,
recall 0.7944 and F1 0.6322. Balanced Random Forest achieved the highest mean
F1 (0.6358). Class weighting consistently increased recall and reduced
precision; the small XGBoost ROC-AUC difference is not treated as decisive.
Retain unweighted XGBoost, balanced XGBoost and balanced Random Forest as
Phase 3 candidates. These are validation results, not held-out test results.

The executed Phase 2A notebook is intentionally committed with outputs as
experimental evidence. Automated tests validate its JSON structure, Python
syntax, saved execution evidence and training-only protocol.

### Phase 3 — Colab results reviewed

The uploaded Phase 3 fold-score, summary and validation-prediction CSVs were reviewed. For sigmoid calibration, balanced XGBoost achieved mean outer-fold F1 0.6359 (precision 0.5560, recall 0.7472), while unweighted XGBoost achieved F1 0.6346 (precision 0.5453, recall 0.7612). The difference is small relative to fold variability and is not evidence of a decisive performance gap. Balanced XGBoost is the provisional Phase 4 finalist because F1 is the primary objective and it has the higher precision; unweighted XGBoost remains the close alternative. The final threshold is still selected from training-only OOF predictions.

### Phase 4 — final evaluation completed

The Phase 4 notebook was executed in Colab and committed to `main`. The final candidate was selected from Phase 3 training-only outer-fold results: **XGBoost (balanced) + sigmoid**. Its final threshold was **0.344207**, selected from training-only OOF probabilities; training OOF F1 was 0.637266. The final base pipeline was fitted on all 5,634 training rows.

| Metric | Threshold 0.50 | Training-OOF threshold 0.344207 |
|---|---:|---:|
| Accuracy | 80.84% | 76.58% |
| Precision | 66.56% | 54.38% |
| Recall | 55.88% | 72.99% |
| F1 | 60.76% | 62.33% |
| ROC-AUC | 84.55% | 84.55% |
| Average Precision | 66.30% | 66.30% |
| Brier score | 0.1356 | 0.1356 |
| Log loss | 0.4156 | 0.4156 |

At threshold 0.344207, the confusion matrix was TN 806, FP 229, FN 101, TP 273. Against the historical Milestone 4 threshold-0.53 reference (accuracy 75.09%, precision 52.02%, recall 79.14%, F1 62.78%, ROC-AUC 84.63%, AP 66.60%), the final model has higher accuracy and precision but lower recall, F1, ROC-AUC and AP. This is a trade-off, not an overall improvement.

The same holdout was used in Milestone 4 and Phase 4. It is not a new independent test set and must not be reused for further tuning or selection. The original Milestone 4 binary is not in Git; this comparison uses the preserved historical metrics. The three CSV exports were generated in Colab's ignored `reports/` directory and were not committed. The executed notebook's aggregate outputs are retained in Git; the row-level prediction CSV is not required for this report.

## Reproducibility and execution

- Supported Python: 3.11–3.12.
- CI Python: 3.12.
- Random seed: 42.
- Automated tests: python -m pytest.
- Use Google Colab or VS Code with the original CSV supplied locally.
- Never use the held-out test split for tuning, feature selection, calibration,
  threshold selection, or experiment decisions.

## Phase 2A — Class-imbalance experiment protocol

- Compare matched balanced and unweighted configurations for Logistic Regression,
  Decision Tree, Random Forest and XGBoost.
- Include Dummy prior as a prevalence-only reference.
- Use existing fixed hyperparameters; only class-weight treatment changes.
- Use the same 5-fold × 3-repeat stratified CV splits for every candidate.
- Keep preprocessing inside each fold through the complete sklearn Pipeline.
- For XGBoost, calculate scale_pos_weight from each fit fold when balanced;
  use 1.0 in the unweighted configuration.
- Compare ROC-AUC, Average Precision, precision, recall, F1, accuracy and fit time.
- Keep the default prediction threshold at 0.50; threshold selection belongs to Phase 3.
- The held-out test partition is reconstructed only to preserve the original split
  and is not passed to any Phase 2A experiment function.
- Phase 2A was executed against the original CSV in Colab; its reviewed results are
  recorded in the Phase 2A results section above.

Notebook: notebooks/06_class_imbalance_experiments.ipynb.

## Current next action

Review and merge PR #13 after its GitHub Actions checks pass. Then run
`notebooks/08_final_model_selection_and_evaluation.ipynb` in Colab using the
original IBM Telco CSV and `phase_3_calibration_threshold_fold_scores.csv`.
Upload the executed notebook and Phase 4 CSV outputs for review. The original
holdout was already used for the Milestone 4 baseline; Phase 4 is the final
frozen-model comparison, and the holdout must not be used for any further tuning.
