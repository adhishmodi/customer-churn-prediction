"""Configuration constants for the Customer Churn Prediction project."""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Model and report paths
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Random seed for reproducibility
RANDOM_SEED = 42

# Target column name
TARGET_COLUMN = "Churn"

# Identifier column (to be excluded from modeling)
ID_COLUMN = "customerID"

# Test set size
TEST_SIZE = 0.2

# Cross-validation folds
CV_FOLDS = 5
