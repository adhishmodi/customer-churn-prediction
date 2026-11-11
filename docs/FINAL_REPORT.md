# Customer Churn Prediction Using Machine Learning

**Academic project report — B.Tech Computer Engineering**

## Abstract

This project develops and evaluates a machine-learning workflow for predicting customer churn using the IBM Telco Customer Churn dataset. The dataset contains 7,043 customer records and has a churn prevalence of 26.54%. The workflow covers exploratory data analysis, leakage-aware preprocessing, baseline model comparison, hyperparameter tuning, class-imbalance experiments, probability calibration, threshold selection, and final evaluation.

The final candidate selected from training-only validation was a class-balanced XGBoost model with sigmoid probability calibration. Its decision threshold of 0.344207 was selected using training-only out-of-fold (OOF) predictions. On the final 1,409-row evaluation partition, it achieved 76.58% accuracy, 54.38% precision, 72.99% recall, 62.33% F1, 84.55% ROC-AUC, and 66.30% Average Precision.

The final model is not uniformly better than the historical Milestone 4 baseline: it improves accuracy and precision at the selected threshold, while recall, F1, ROC-AUC, and Average Precision are slightly lower. The original holdout had already been used for the Milestone 4 baseline, so the final results are a controlled historical comparison and not an independent external validation.

## 1. Introduction

Customer churn occurs when a customer stops using a company's service. Identifying customers who may leave can help a telecom provider prioritize retention activities. A binary classification model can estimate the probability that a customer belongs to the churn class using available customer, service, contract, and billing information.

### 1.1 Objectives

- Explore the customer dataset and identify data-quality issues and observed churn patterns.
- Build a reproducible preprocessing and model-training pipeline.
- Compare classification algorithms using cross-validation.
- Examine the effect of class weighting on precision and recall.
- Evaluate probability calibration and training-only decision-threshold selection.
- Freeze a final candidate before evaluating the historical holdout.
- Document performance honestly, including trade-offs and limitations.

## 2. Dataset description

The project uses the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

| Property | Value |
|---|---:|
| Customer records | 7,043 |
| Original columns | 21 |
| Identifier | `customerID` |
| Prediction target | `Churn` |
| Predictor count after excluding identifier and target | 19 |
| Numeric predictors | 3 |
| Categorical predictors | 16 |
| Non-churn customers | 5,174 (73.46%) |
| Churn customers | 1,869 (26.54%) |

The target is binary: `No` is encoded as 0 and `Yes` as 1. The identifier `customerID` is excluded from model features.

### 2.1 Exploratory data analysis

The EDA notebook recorded no duplicate rows. The `TotalCharges` column contained 11 blank values associated with zero-tenure customers; these were converted to zero in the cleaned analysis copy. The raw input dataframe was preserved.

Observed category-level churn rates included:

| Category | Observed churn rate |
|---|---:|
| Month-to-month contract | 42.71% |
| Electronic check payment | 45.29% |
| Fiber optic internet | 41.89% |
| No technical support | 41.64% |
| No online security | 41.77% |

The numeric correlation between tenure and the binary churn target was -0.352 in the EDA output. These are descriptive associations in this dataset; they do not establish that any feature causes churn.

## 3. Data preprocessing

The original data was cleaned on a separate copy. The deterministic steps were:

- Convert `TotalCharges` to numeric values.
- Replace missing `TotalCharges` with zero only for zero-tenure customers.
- Convert `SeniorCitizen` from 0/1 to readable categories.
- Remove `customerID`.
- Encode `Churn` as 0/1.

The data was split using a stratified 80/20 split with random seed 42.

| Partition | Rows | Churn prevalence |
|---|---:|---:|
| Training | 5,634 | 26.54% |
| Held-out evaluation | 1,409 | 26.54% |
| Total | 7,043 | 26.54% |

The preprocessing pipeline was fitted on training data only:

| Feature type | Missing-value handling | Transformation |
|---|---|---|
| Numeric | Median imputation | Standard scaling |
| Categorical | Most-frequent imputation | One-hot encoding |

Unknown categories are ignored during one-hot transformation. The preprocessing transformer is part of the estimator pipeline so that every cross-validation fold learns preprocessing statistics only from its own training rows. The preprocessing notebook recorded 30 encoded features after transformation.

## 4. Model training and validation methodology

The initial candidate set included a Dummy prior classifier, Logistic Regression, Decision Tree, Random Forest, and XGBoost. Five-fold shuffled StratifiedKFold with seed 42 was used. Mean cross-validation ROC-AUC was the primary model-selection metric; Average Precision and threshold-based metrics were also recorded because churn is the minority class.

RandomizedSearchCV was used to tune Logistic Regression, Random Forest, and XGBoost, with ten sampled configurations per tuned model in the recorded training notebook. The held-out partition was not supplied to model fitting or hyperparameter selection.

### 4.1 Initial and tuned model results

The recorded tuned-model comparison was:

| Tuned candidate | Mean CV ROC-AUC | ROC-AUC std | Mean CV Average Precision |
|---|---:|---:|---:|
| Tuned XGBoost | **0.8501** | 0.0119 | **0.6694** |
| Tuned Random Forest | 0.8467 | 0.0106 | 0.6613 |
| Tuned Logistic Regression | 0.8462 | 0.0128 | 0.6592 |

Tuned XGBoost was selected as the historical Milestone 3 candidate because it had the highest mean CV ROC-AUC.

## 5. Milestone 5 experiments

### 5.1 Repeated cross-validation and class imbalance

The class-imbalance experiment compared matched balanced and unweighted configurations for Logistic Regression, Decision Tree, Random Forest, and XGBoost. A Dummy prior classifier was included as a reference. Each candidate used the same 5-fold × 3-repeat stratified CV splits, for 15 training-only validation folds per candidate. The held-out test partition was not used.

Selected results:

| Candidate | Mean ROC-AUC | Mean Average Precision | Mean precision | Mean recall | Mean F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost (unweighted) | **0.8490** | **0.6701** | 0.6709 | 0.5233 | 0.5873 |
| XGBoost (balanced) | 0.8484 | 0.6678 | 0.5254 | **0.7944** | 0.6322 |
| Random Forest (balanced) | 0.8449 | 0.6579 | 0.5655 | 0.7273 | **0.6358** |

Class weighting increased recall and reduced precision across the compared model families. It did not consistently improve ROC-AUC or Average Precision. The XGBoost ROC-AUC difference between weighted and unweighted variants was small relative to fold variability. The experiment therefore retained unweighted XGBoost, balanced XGBoost, and balanced Random Forest for the calibration stage rather than claiming that class weighting was universally superior.

### 5.2 Probability calibration and threshold optimization

The next experiment compared no calibration, sigmoid calibration, and isotonic calibration for three candidates: unweighted XGBoost, balanced XGBoost, and balanced Random Forest.

A nested 5-fold × 5-fold stratified validation procedure was used on the original training partition. Inner OOF predictions from each outer-training partition fitted the calibrator and selected an F1-maximizing threshold. The complete calibrated procedure was then evaluated on the untouched outer-validation fold. The project holdout was not used in this experiment.

For sigmoid calibration, the two XGBoost candidates had the following outer-fold summaries:

| Candidate | Mean F1 | F1 std | Mean precision | Mean recall |
|---|---:|---:|---:|---:|
| XGBoost (balanced) + sigmoid | **0.635881** | 0.024157 | **0.555974** | 0.747157 |
| XGBoost (unweighted) + sigmoid | 0.634574 | 0.019293 | 0.545255 | **0.761204** |

The F1 difference was small. Balanced XGBoost had slightly higher mean F1 and precision, while unweighted XGBoost had higher recall. Balanced XGBoost was therefore the provisional finalist under the agreed F1-primary, precision-tie-break objective. This was not evidence of a decisive performance gap.

### 5.3 Finalist selection and threshold freeze

Phase 4 selected the final candidate using the Phase 3 outer-fold training-only results. The selection order was mean F1, then mean precision, then mean recall.

The selected configuration was:

- Classifier: XGBoost (balanced)
- Calibration: sigmoid
- Final decision threshold: **0.344207**
- Threshold selection: maximize F1 using training-only OOF calibrated probabilities
- Training-only OOF F1 at the selected threshold: **0.637266**
- Final fitting rows: 5,634

The model, calibrator, and threshold were frozen before final evaluation. No test features or labels were passed to the fitting or threshold-selection function.

## 6. Final evaluation

The final model was evaluated on 1,409 rows at the conventional 0.50 threshold and at the threshold selected from training-only OOF predictions.

### 6.1 Final model metrics

| Metric | Threshold 0.50 | Training-OOF threshold 0.344207 |
|---|---:|---:|
| Accuracy | 80.84% | 76.58% |
| Precision | 66.56% | 54.38% |
| Recall | 55.88% | 72.99% |
| F1-score | 60.76% | **62.33%** |
| ROC-AUC | 84.55% | 84.55% |
| Average Precision | 66.30% | 66.30% |
| Brier score | 0.1356 | 0.1356 |
| Log loss | 0.4156 | 0.4156 |

The threshold changes the binary classifications, so accuracy, precision, recall, F1, and the confusion matrix change. ROC-AUC and Average Precision are based on probability ranking and therefore remain the same across these two thresholds.

### 6.2 Confusion matrices

At threshold 0.50:

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 930 | 105 |
| Actual churn | 165 | 209 |

At threshold 0.344207:

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 806 | 229 |
| Actual churn | 101 | 273 |

The lower threshold identifies more actual churners (higher recall) but also flags more non-churn customers (more false positives). This is the expected precision-recall trade-off.

## 7. Comparison with the historical Milestone 4 baseline

The historical baseline was Tuned XGBoost. Its original held-out evaluation at threshold 0.50 was accuracy 74.02%, precision 50.67%, recall 81.02%, F1 62.35%, ROC-AUC 84.63%, and Average Precision 66.60%.

Milestone 4 also recorded a training-only OOF threshold of 0.53. Its historical test metrics were accuracy 75.09%, precision 52.02%, recall 79.14%, F1 62.78%, ROC-AUC 84.63%, and Average Precision 66.60%.

The comparison against the threshold-0.53 historical reference is:

| Metric | Historical baseline (0.53) | Final model (0.344207) |
|---|---:|---:|
| Accuracy | 75.09% | **76.58%** |
| Precision | 52.02% | **54.38%** |
| Recall | **79.14%** | 72.99% |
| F1-score | **62.78%** | 62.33% |
| ROC-AUC | **84.63%** | 84.55% |
| Average Precision | **66.60%** | 66.30% |

The final model improves accuracy by 1.49 percentage points and precision by 2.36 percentage points relative to that historical threshold reference. It has lower recall by 6.15 percentage points, lower F1 by 0.45 percentage points, lower ROC-AUC by 0.08 percentage points, and lower Average Precision by 0.30 percentage points.

**Conclusion from the comparison:** the final candidate changes the operating trade-off; it does not establish an overall predictive-performance improvement. The preferred threshold depends on the relative cost of missed churners and unnecessary retention contacts.

The exact original Milestone 4 model binary is not stored in the Git repository, so the Phase 4 comparison uses the preserved historical metrics rather than a direct binary-to-binary comparison.

## 8. Limitations

1. **Holdout reuse:** the same 1,409-row partition was used in the historical Milestone 4 evaluation and the Phase 4 comparison. The Phase 4 result is not a fresh independent test.
2. **Single dataset:** the evaluation uses one public telecom dataset. Performance may differ for other providers, customer populations, or time periods.
3. **No prospective validation:** no future-period or live customer data was evaluated.
4. **Business costs not modeled:** F1 balances precision and recall but does not encode the monetary cost of false positives, missed churners, or retention offers.
5. **Deferred experiments:** feature-engineering Phase 2B and further hyperparameter-optimization Phase 2C were deferred. The project does not claim that every possible feature or tuning strategy was explored.
6. **No causal conclusions:** observed category-level churn rates and feature correlations are associations, not causal effects.
7. **Historical artifact limitation:** the original baseline metrics are preserved, but the original model binary is unavailable in the repository for a direct artifact comparison.

Because the holdout has now been used for final comparison, it must not be used for further tuning, threshold selection, feature selection, or model selection. Future evaluation should use training-only cross-validation or a genuinely new external dataset.

## 9. Conclusion

This project implemented an end-to-end, leakage-aware churn-prediction workflow. It established a tuned XGBoost historical baseline, evaluated class weighting under repeated cross-validation, compared calibration methods using nested validation, and froze a balanced XGBoost model with sigmoid calibration and a training-only F1 threshold.

The final configuration achieved 62.33% F1, 72.99% recall, and 54.38% precision on the final comparison partition. Compared with the historical threshold-0.53 baseline, it improved accuracy and precision but had slightly lower recall, F1, ROC-AUC, and Average Precision. The result should therefore be presented as a different precision-recall operating point, not as an unqualified improvement.

The project demonstrates why model selection, preprocessing, calibration, and threshold optimization must be separated from final evaluation. For a real retention program, the next evaluation should incorporate business costs and an independent future-period dataset.

## 10. Reproducibility

- Dataset filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Random seed: 42
- Train/holdout split: stratified 80/20
- Training rows: 5,634
- Holdout rows: 1,409
- Supported Python: 3.11–3.12
- CI Python: 3.12
- Automated test command: `python -m pytest`
- Core dependencies: NumPy, pandas, scikit-learn, XGBoost, joblib, matplotlib, seaborn, and pytest

The executed notebooks are retained under `notebooks/`. Generated model and report artifacts are stored in ignored local `models/` and `reports/` directories and are not automatically committed.

## References

1. IBM Telco Customer Churn dataset, hosted on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Project implementation and executed experiment notebooks: https://github.com/adhishmodi/customer-churn-prediction
