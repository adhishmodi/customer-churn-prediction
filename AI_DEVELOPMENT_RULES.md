# AI Development Rules

This document defines the standards and practices for developing the Customer Churn Prediction System.

## 1. General Principles

### 1.1 No Fabrication
- Never invent data, statistics, or model results.
- Report actual outputs from code execution.
- If a result is unexpected, investigate rather than assume.

### 1.2 Reproducibility
- Use fixed random seeds (42) for all stochastic operations.
- Document all hyperparameters and preprocessing decisions.
- Version control all code and configuration.

### 1.3 Data Integrity
- Preserve raw data in data/raw/ -- never modify it.
- Create processed copies in data/processed/.
- Document all transformations applied.

### 1.4 No Data Leakage
- Split data before any preprocessing/fitting.
- Fit transformers (scalers, encoders) on training data only.
- Apply fitted transformers to test data.
- Never use target information during feature engineering.

## 2. Code Organization

### 2.1 Modular Structure
src/
|-- __init__.py
|-- config.py           # Configuration constants
|-- data_loader.py      # Dataset loading utilities
|-- validate.py         # Data validation checks
|-- preprocess.py       # Preprocessing pipeline
|-- features.py         # Feature engineering
|-- train.py            # Model training
|-- evaluate.py         # Model evaluation
|-- explain.py          # SHAP explainability
|-- app.py              # Streamlit application

### 2.2 Naming Conventions
- Files: snake_case.py
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: UPPER_SNAKE_CASE
- Private: _leading_underscore

### 2.3 Type Hints
- Use type hints for all public functions.
- Use from __future__ import annotations for forward references.

## 3. Machine Learning Practices

### 3.1 Random Seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

### 3.2 Train-Test Split
- Use stratified split for classification.
- Typical split: 80/20 or 70/30.
- Store split indices for reproducibility.

### 3.3 Preprocessing Pipeline
- Use sklearn.pipeline.Pipeline and ColumnTransformer.
- Separate numeric and categorical preprocessing.
- Handle missing values explicitly.

### 3.4 Model Evaluation
- Primary metric: ROC-AUC (handles class imbalance).
- Secondary metrics: Precision, Recall, F1, Accuracy.
- Use cross-validation for robust estimates.
- Report confidence intervals where possible.

### 3.5 Class Imbalance
- Churn rate ~26% (imbalanced).
- Use class_weight=balanced or scale_pos_weight.
- Evaluate on precision-recall curve, not just ROC.

## 4. Documentation Standards

### 4.1 Docstrings (NumPy style)
def function_name(param: type) -> return_type:
    """Short description.

    Longer description if needed.

    Parameters
    ----------
    param : type
        Description.

    Returns
    -------
    return_type
        Description.
    """

### 4.2 Inline Comments
- Explain why, not what.
- Document non-obvious decisions.
- Reference external sources for techniques.

## 5. Testing

### 5.1 Unit Tests
- Test data loading and validation functions.
- Test preprocessing pipeline components.
- Test model training and evaluation functions.
- Use pytest with fixtures.

### 5.2 Test Coverage
- Aim for >80% coverage on core modules.
- Test edge cases (empty data, missing values, etc.).

## 6. Git Practices

### 6.1 Commits
- Atomic commits with descriptive messages.
- Format: type(scope): subject
- Types: feat, fix, docs, refactor, test, chore

### 6.2 Branching
- main: Stable, deployable code.
- feature/*: New features.
- bugfix/*: Bug fixes.

## 7. Batch Discipline

### 7.1 Sequential Execution
- Complete each batch fully before starting the next.
- Do not implement future batch functionality prematurely.
- Wait for explicit instruction to proceed.

### 7.2 Batch Deliverables
Each batch must produce:
- Working code that passes tests.
- Updated PROJECT_STATUS.md.
- Clear next-step recommendation.

## 8. ML Concept Explanations

When implementing key ML concepts, include brief explanations:
- Why this technique? (theory)
- Why these hyperparameters? (practical choice)
- What are the trade-offs? (bias-variance, interpretability, etc.)

## 9. Error Handling

- Validate inputs early with clear error messages.
- Use logging for debugging, not print statements.
- Fail fast on data quality issues.

## 10. Performance

- Profile before optimizing.
- Use vectorized operations (pandas/numpy).
- Consider memory usage for large datasets.
