# Project Status: Customer Churn Prediction System

## Current Batch: Batch 3 — Data Preprocessing & Feature Engineering

### Status: COMPLETED

### Batch 1 Completed:
- [x] Project directory structure created
- [x] Virtual environment created (.venv)
- [x] requirements.txt with all dependencies
- [x] .gitignore configured
- [x] README.md created
- [x] PROJECT_STATUS.md created
- [x] AI_DEVELOPMENT_RULES.md created
- [x] Core dependencies installed (pandas, numpy, matplotlib, seaborn, scikit-learn)
- [x] Additional dependencies installed (xgboost, shap, streamlit, pytest)
- [x] Dataset available in data/raw/
- [x] src/config.py, src/data_loader.py, src/validate.py created
- [x] Dataset verified: 7,043 rows x 21 columns, 0 duplicates, 0 missing (after TotalCharges fix)

### Batch 2 Completed:
- [x] Customer demographics analysis (gender, SeniorCitizen, Partner, Dependents)
- [x] Churn distribution analysis (26.5% churn rate)
- [x] Tenure analysis (strongest predictor: r = -0.352)
- [x] Contract and payment analysis (Month-to-month: 42.7%, Electronic check: 45.3%)
- [x] Monthly and Total Charges analysis (Churners pay \ more/month)
- [x] Correlation and feature relationships (heatmap generated)
- [x] Internet & phone services analysis (Fiber optic: 41.9%, No security: 41.8%)
- [x] Paperless billing analysis (33.6% vs 16.3%)
- [x] 9 visualization charts saved to reports/figures/
- [x] EDA report created: reports/EDA_REPORT.md
- [x] Reusable EDA module: src/eda.py

### Batch 3 Completed:
- [x] Handle missing and invalid values (TotalCharges converted, 11 blanks filled with median)
- [x] Remove irrelevant identifiers (customerID dropped)
- [x] Encode categorical variables (OneHotEncoder with drop='first', 16 categorical -> 27 features)
- [x] Scale numerical variables (StandardScaler on tenure, MonthlyCharges, TotalCharges)
- [x] Handle class imbalance (class_weight computed: {0: 0.68, 1: 1.88}, scale_pos_weight=2.77)
- [x] Split training and testing data without leakage (stratified 80/20, random_state=42)
- [x] Build reusable preprocessing pipelines (ColumnTransformer saved as preprocessor.joblib)
- [x] Save processed datasets to data/processed/ (X_train.npy, X_test.npy, y_train.npy, y_test.npy, feature_names.txt)

### Preprocessing Results:
- **Input features**: 19 (after dropping customerID)
- **Output features**: 30 (3 numeric scaled + 27 one-hot encoded)
- **Train set**: 5,634 samples (79.99%)
- **Test set**: 1,409 samples (20.01%)
- **Stratification verified**: Train churn rate 26.54%, Test churn rate 26.54%
- **No data leakage**: Preprocessor fitted ONLY on training data
- **No NaN values**: All missing handled

### Pending (Batch 4):
- [ ] Train baseline models (Logistic Regression, Random Forest, XGBoost)
- [ ] Model evaluation with ROC-AUC, Precision, Recall, F1
- [ ] Cross-validation for robust estimates

### Pending (Batch 5):
- [ ] Hyperparameter tuning
- [ ] Model comparison and selection

### Pending (Batch 6):
- [ ] SHAP explainability
- [ ] Streamlit application
- [ ] Final documentation

---

## Environment Details:
- Python: 3.14.6 (in .venv)
- OS: Windows
- Dataset: IBM Telco Customer Churn (7,043 records, 21 columns)
- Target: Churn (Yes/No)

## Key Files:
- data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv (977 KB)
- data/processed/X_train.npy, X_test.npy, y_train.npy, y_test.npy
- data/processed/preprocessor.joblib, feature_names.txt
- requirements.txt (updated with xgboost, shap, streamlit, pytest)
- .gitignore (excludes venv, data, models, reports)
- src/__init__.py, src/config.py, src/data_loader.py, src/validate.py, src/eda.py, src/preprocess.py
- reports/figures/01-09_*.png (9 visualization charts)
- reports/EDA_REPORT.md

## Next Action:
Start Batch 4 — Baseline Model Training & Evaluation
