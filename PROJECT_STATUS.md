# Project Status — Customer Churn Prediction

## Current state

**Milestone 1 — Exploratory data analysis: completed and merged.**

**Milestone 2 — Data preprocessing: completed and merged.**

**Milestone 3 — Model training and selection: completed and merged.**

**Milestone 4 — Model evaluation: completed and merged.**

**Milestone 5 — Model improvement, optimization, final inference artifact, and application: academic phases complete; Phase 7C deployment implementation merged and service live, with hosted model inference unverified due to a secrets-file parsing error.** Phases 0–1, 2A, 2B, 2C, 3, 4, 5 and 6 are merged. Phase 2C was merged in PR #17. Phase 6 strict nested cross-fitting compared tuned unweighted XGBoost with the fixed Phase 4 balanced-XGBoost candidate using training-only validation; results were reviewed and PR #18 was merged. Phase 7A builds the deployable balanced-XGBoost + sigmoid artifact and inference interface. The historical holdout is permanently excluded from all Phase 7A fitting, calibration, threshold selection, and evaluation.

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

- [x] Phase 2A: controlled class-imbalance experiments (PR #10 merged; Colab run completed; results reviewed).
- [x] Phase 2B: feature-engineering experiments (corrected feature routing merged in PR #16; corrected Colab results reviewed; original features retained).
- [x] Phase 2C: nested hyperparameter optimization (PR #17 merged; executed Colab notebook and result evidence committed).
- [x] Phase 3: nested probability calibration and F1-threshold optimization implemented (PR #12 merged); Colab results reviewed.
- [x] Phase 4: final training-only finalist selection, calibration and threshold freeze, then final comparison against the historical holdout (PR #13 merged; executed notebook committed).
- [x] Phase 5: academic documentation and final report (PR #14 merged).

### Phase 6 — Calibrated finalist comparison (completed)

- [x] Implement strict nested cross-fitting for tuned unweighted XGBoost and the fixed Phase 4 balanced-XGBoost candidate (PR #18).
- [x] Fit sigmoid calibration and select each fold's F1 threshold from outer-training OOF predictions only.
- [x] Compare both candidates on identical five-fold outer validation splits; report fold-level metrics, paired fold deltas, pooled OOF metrics, thresholds, and selected parameters.
- [x] Run focused and full automated tests; verify the Phase 6 notebook structure (GitHub Actions: 77 passed).
- [x] Run the Phase 6 notebook in Colab and review the training-only finalist comparison outputs.

**Permanent exclusion rule:** the historical holdout is not an input to Phase 6. The Phase 6 API accepts only `X_train` and `y_train`; notebook split reconstruction discards the holdout outputs immediately. No Phase 6 tuning, calibration, threshold selection, or candidate decision may use historical holdout data. Five outer folds are descriptive and must not be presented as a formal significance test.

### Phase 2B — Feature-engineering experiment implementation

The Phase 2B implementation adds four controlled feature configurations:
original predictors, tenure groups, subscribed-service count, and both
engineered features together. The tenure bands are 0–12, 13–24, 25–48,
49–60, and 61+ months. Service count sums the eight binary service indicators
(PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, and StreamingMovies), counting only exact "Yes"
values. Original predictors are retained in every configuration.

The three Phase 2A finalists (unweighted XGBoost, balanced XGBoost, and
balanced Random Forest) are compared across all four configurations using
identical 5-fold × 3-repeat stratified CV (180 fold-level candidate results).
Feature transformations and learned preprocessing are contained in each
scikit-learn Pipeline. The experiment API accepts only X_train and y_train;
the historical holdout is reconstructed only to preserve the split and is
discarded before experimentation.

Notebook: notebooks/09_feature_engineering.ipynb.

The initial Colab run produced identical metrics across feature configurations because the ColumnTransformer dropped both engineered columns. Those results were invalid and superseded. PR #16 corrected routing, added classifier-input regression tests, and added a notebook preflight audit. The corrected 180-row run was reviewed: original features achieved the strongest ROC-AUC in the two XGBoost families and won all three repeat-level ROC-AUC comparisons. No engineered configuration showed a consistent improvement, so Phase 2C retains original features. The historical holdout was not used.

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

## Phase 7A — final model artifact and inference (completed)

- [x] Re-audit the latest `main`, confirm the final notebook fix (PR #24), and verify the post-merge repository state.
- [x] Build the deployable artifact using only the original 5,634-row training partition reconstructed with the seed-42 stratified split.
- [x] Fit five-fold training-only out-of-fold sigmoid calibration for the Phase 4 balanced-XGBoost candidate.
- [x] Select a fresh F1 threshold from calibrated training-only OOF predictions; do not reuse the historical Phase 4 threshold.
- [x] Save and verify the versioned model bundle and JSON metadata.
- [x] Reload the serialized artifact and pass synthetic single-customer and batch-inference smoke tests in Colab.
- [x] Record the executed evidence in `notebooks/12_phase_7a_final_model_training.ipynb`.

### Verified Colab execution

Notebook 12 was executed and saved on `main`. Its recorded source commit is `79ae02d61c2fb4bbd994cc70db685aa7c0291209`.

| Item | Verified result |
|---|---:|
| Original dataset preflight | 7,043 rows × 21 columns |
| Training partition | 5,634 rows |
| Historical holdout | 1,409 rows, excluded from model fitting and evaluation |
| Final candidate | Balanced XGBoost + sigmoid calibration |
| Calibration | 5-fold training-only OOF |
| Fresh threshold | 0.3442067252 |
| Training-OOF F1 at selected threshold | 0.6372663551 |
| Serialized model artifact | 57,872 bytes |
| Metadata verification | Passed |
| Artifact reload and synthetic single/batch inference | Passed |

The training-OOF F1 is the score used to select the decision threshold. It is **not** an independent estimate of generalization performance and must not be presented as a fresh test score.

### Artifact handling and holdout restriction

The generated local files are:

- `models/customer_churn_model.joblib` — fitted preprocessing/model pipeline, sigmoid calibrator, decision threshold, schema, and provenance.
- `models/training_metadata.json` — dataset/split provenance, calibration and threshold details, package versions, and artifact size.

Both files are generated locally by the Colab notebook and remain excluded from Git. The existing `.gitignore` excludes `models/*` and `*.joblib`; the metadata JSON is inside the ignored `models/` directory. Do not commit either file or the raw CSV.

**Permanent holdout exclusion:** the original 1,409-row historical holdout is reconstructed only to preserve the original split and is discarded before cleaning, fitting, calibration, threshold selection, inference validation, or evaluation. Its labels are used only for stratification. No historical-holdout metrics are calculated in Phase 7A, and the holdout must never be reused.

Notebook: `notebooks/12_phase_7a_final_model_training.ipynb`.




### Phase 2B — corrected results reviewed

The feature-routing correction was merged in PR #16. The corrected experiment compared the three Phase 2A finalists across the four planned feature configurations using identical 5-fold × 3-repeat training-only CV. The regenerated outputs contained 180 fold results across 12 candidates. Predictive metrics varied across feature configurations, confirming that the engineered columns reached the classifier. Original features achieved the strongest ROC-AUC in both XGBoost variants and won the repeat-level ROC-AUC comparison in all three repeats. No engineered configuration demonstrated a consistent improvement. Retain the original feature configuration for Phase 2C. The earlier run in which all feature configurations had identical metrics remains invalid and must not be used for feature selection.

### Phase 2C — nested hyperparameter optimization implementation

The Phase 2C implementation tunes the three Phase 2A finalists using the original feature configuration:

- XGBoost (unweighted)
- XGBoost (balanced)
- Random Forest (balanced)

The protocol uses 5 outer stratified folds and 3 inner stratified folds, with 12 randomized parameter combinations per candidate per outer fold (540 inner model fits). ROC-AUC is the predeclared primary selection metric. Average Precision, precision, recall, F1 and accuracy are secondary metrics. Binary classification metrics use the default 0.50 threshold; threshold optimization is outside this phase.

For each outer fold, all candidates are tuned on the outer training rows using identical inner folds. The candidate with the highest inner-CV ROC-AUC is evaluated once on the untouched outer validation fold. Preprocessing remains inside the pipeline; XGBoost class weighting and Random Forest class weights are computed within each model fit. The experiment API accepts only X_train and y_train and records test_set_used=False. The outer-fold summary estimates the complete tuning-and-selection procedure; inner-CV scores are not reported as unbiased generalization estimates.

Notebook: notebooks/10_hyperparameter_optimization.ipynb.

Phase 2C was completed in Colab and merged in PR #17 with executed notebook evidence. Five outer folds provide limited stability evidence; standard deviations are descriptive and must not be presented as formal confidence intervals or significance tests. The historical holdout remains permanently excluded.


## Phase 7B — interactive Streamlit prediction app (completed)

### Scope and implementation
- [x] Add a Streamlit entry point that loads the trusted Phase 7A artifact through the existing inference module.
- [x] Add a single-customer form using the exact 19 model features.
- [x] Display calibrated churn probability, the artifact's Yes/No decision, and its stored threshold.
- [x] Add batch CSV validation and prediction with bounded upload size and row count.
- [x] Add downloadable prediction output and neutralize formula-like text values during CSV export.
- [x] Keep customer inputs in memory only; do not persist uploaded data.
- [x] Add focused unit tests for input parsing, schema validation, batch limits/order, and CSV export safety.
- [x] Add Notebook 13 for an independent Colab runtime using the trusted artifact downloaded from Notebook 12.
- [x] Merge the Colab runtime and requirements-path fixes (PRs #27 and #28).
- [x] Pass the GitHub Actions test workflow.
- [x] Complete a live Colab smoke test using the locally generated Phase 7A artifact.
- [x] Complete manual batch-upload validation: valid CSV produced predictions; missing-feature, extra-column, and invalid-numeric CSVs were rejected as expected.

### Verified application behavior
- Live Streamlit app launched successfully in Google Colab through the Colab port proxy.
- The artifact loaded and produced single-customer and batch predictions.
- Observed example outputs were internally consistent with the stored threshold (0.3442067252): 0.68836 predicted Yes and 0.04088 predicted No.
- Manual CSV tests matched expected behavior for all four supplied files.
- GitHub Actions run #43703636079 completed successfully.
- These checks validate application/inference integration and input handling; they are not a new estimate of model generalization performance.

### Boundaries and safeguards
Phase 7B reuses the Phase 7A model and inference code. It does not retrain, tune, recalibrate, select a new threshold, or calculate evaluation metrics. It does not add a database, store customer records, accept arbitrary uploaded model binaries, or deploy the app to a public hosting provider.

The model artifact and metadata remain local and Git-ignored. Notebook 13 supports separate Colab runtimes: download the `.joblib` artifact from Notebook 12, then upload that trusted artifact when Notebook 13 prompts for it. Notebook 13 validates the bundle with `load_model_artifact()` before starting Streamlit. Because joblib uses pickle-based deserialization, only upload the artifact generated by your own Notebook 12 run; never upload an untrusted file. The historical 1,409-row holdout remains permanently excluded and is not used by this app phase. A prediction app smoke test verifies application behavior only; it is not a model-performance evaluation.

### Phase 7B completion decision
Phase 7B is functionally complete based on the merged implementation, successful CI, successful live Colab launch, prediction output, and four manual CSV validation cases. No public deployment is claimed.


## Phase 7C — Render deployment (implementation)

### Approved scope
- [x] Use a Render Python Web Service in the confirmed workspace, starting on the Free plan.
- [x] Keep the app public and unauthenticated; do not add a database.
- [x] Load the trusted Phase 7A model through Streamlit Secrets backed by a Render Secret File.
- [x] Keep the model artifact, Base64 payload, checksum, and local secrets file out of Git.
- [x] Preserve existing single-customer and batch prediction behavior, schema checks, upload limits, row limits, and CSV formula mitigation.
- [x] Pin the hosted runtime to Python 3.12.
- [x] Defer the separate `TotalCharges` numeric-validation change.
- [x] Merge the Render deployment implementation in PR #31.
- [x] Verify the Render service reached Live on the expected main-branch commit.
- [ ] Complete hosted model-inference verification: Render logs reported a TOML parsing error in `/etc/secrets/secrets.toml`; the model secrets may not be available to Streamlit.
- [ ] Optional follow-up only: correct the private secret-file formatting and run hosted prediction smoke tests. This is not a blocker for academic project completion.

### Model and data boundaries
Phase 7C serves the existing Phase 7A artifact only. It does not train,
tune, recalibrate, change the decision threshold, or evaluate model
performance. The historical 1,409-row holdout remains permanently excluded.
The application does not persist uploaded customer rows, but the hosting
provider's runtime processes the uploaded data. Because the app is public and
has no authentication or rate limiter, it is a portfolio demonstration only;
use synthetic or de-identified data.

The Render Free web service may spin down after inactivity and can have a cold
start on the next request. It is suitable for a portfolio demonstration, not
a production customer-facing service with uptime, access-control, or capacity
requirements.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the operational setup and
privacy checklist.


## Final project closure — 2026-11-15

**Academic ML project: complete.** The project includes EDA, preprocessing, baseline model comparison, class-imbalance experiments, corrected feature-engineering experiments, nested hyperparameter optimization, nested calibration and threshold selection, strict training-only finalist comparison, final training-only model artifact generation, and a Streamlit inference application. Phase 2C was merged in PR #17; Phase 6 was merged in PR #18; the Render deployment implementation was merged in PR #31.

**Final candidate/artifact:** balanced XGBoost with sigmoid calibration, trained using the original training partition and a fresh training-only OOF threshold of 0.3442067252. The historical 1,409-row holdout is permanently excluded from all future development, tuning, calibration, threshold selection, and evaluation. Do not generate new metrics from it.

**Deployment:** Render service `customer-churn-prediction` is Live at https://customer-churn-prediction-a9wc.onrender.com. This confirms service deployment, not successful model loading. Runtime logs reported `Failed to load secrets` because `/etc/secrets/secrets.toml` could not be parsed (illegal newline in the file at line 1, column 77187). Hosted inference and hosted smoke tests are therefore unverified. No further deployment work is required for project closure.

**Known deferred item:** `TotalCharges` numeric validation remains unchanged and out of scope. No code, model, dataset, or application behavior is changed by this final documentation update.
