"""Data preprocessing and feature engineering for Customer Churn Prediction."""
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

from .config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, RANDOM_SEED, 
    TARGET_COLUMN, ID_COLUMN, TEST_SIZE
)
from .data_loader import load_raw_data

# Column definitions based on EDA
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

DROP_FEATURES = [ID_COLUMN]

def load_and_clean_data() -> pd.DataFrame:
    df = load_raw_data()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mask_tenure_zero = df["tenure"] == 0
    df.loc[mask_tenure_zero, "TotalCharges"] = df.loc[mask_tenure_zero, "TotalCharges"].fillna(0)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df = df.drop(columns=[ID_COLUMN], errors="ignore")
    return df

def get_preprocessing_pipeline() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor

def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    return preprocessor.get_feature_names_out().tolist()

def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_SEED, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_param)
    return X_train, X_test, y_train, y_test

def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    preprocessor = get_preprocessing_pipeline()
    preprocessor.fit(X_train)
    return preprocessor

def transform_data(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(X)

def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print("  Saved preprocessor: " + str(path))

def load_preprocessor(path: Path) -> ColumnTransformer:
    return joblib.load(path)

def save_processed_data(X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_test: pd.Series, feature_names: List[str], output_dir: Path = PROCESSED_DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train.values)
    np.save(output_dir / "y_test.npy", y_test.values)
    with open(output_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))
    pd.DataFrame(X_train, columns=feature_names).to_csv(output_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test, columns=feature_names).to_csv(output_dir / "X_test.csv", index=False)
    pd.DataFrame({"Churn": y_train.values}).to_csv(output_dir / "y_train.csv", index=False)
    pd.DataFrame({"Churn": y_test.values}).to_csv(output_dir / "y_test.csv", index=False)
    print("  Saved processed data to: " + str(output_dir))
    print("    X_train: " + str(X_train.shape))
    print("    X_test: " + str(X_test.shape))
    print("    y_train: " + str(y_train.shape))
    print("    y_test: " + str(y_test.shape))
    print("    Features: " + str(len(feature_names)))

def run_preprocessing() -> dict:
    print("=" * 60)
    print("STARTING DATA PREPROCESSING (BATCH 3)")
    print("=" * 60)
    print("\n1. Loading and cleaning data...")
    df = load_and_clean_data()
    print("   Cleaned data shape: " + str(df.shape))
    print("   Missing values: " + str(df.isna().sum().sum()))
    print("\n2. Splitting features and target...")
    X, y = prepare_features_target(df)
    print("   Features: " + str(X.shape[1]) + ", Target distribution: " + str(y.value_counts().to_dict()))
    print("\n3. Performing stratified train/test split...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("   Train: " + str(X_train.shape[0]) + " (" + str(X_train.shape[0]/len(X)*100) + "%)")
    print("   Test:  " + str(X_test.shape[0]) + " (" + str(X_test.shape[0]/len(X)*100) + "%)")
    print("   Train churn rate: " + str(y_train.mean()*100) + "%")
    print("   Test churn rate:  " + str(y_test.mean()*100) + "%")
    print("\n4. Fitting preprocessing pipeline on training data...")
    preprocessor = fit_preprocessor(X_train)
    print("\n5. Transforming train and test data...")
    X_train_processed = transform_data(preprocessor, X_train)
    X_test_processed = transform_data(preprocessor, X_test)
    feature_names = get_feature_names_out(preprocessor)
    print("   Processed features: " + str(len(feature_names)))
    print("   X_train shape: " + str(X_train_processed.shape))
    print("   X_test shape:  " + str(X_test_processed.shape))
    print("\n6. Saving preprocessor and processed data...")
    save_preprocessor(preprocessor, PROCESSED_DATA_DIR / "preprocessor.joblib")
    save_processed_data(X_train_processed, X_test_processed, y_train, y_test, feature_names)
    print("\n7. Calculating class weights for imbalance handling...")
    class_counts = y_train.value_counts()
    n_samples = len(y_train)
    n_classes = len(class_counts)
    class_weights = {cls: n_samples / (n_classes * count) for cls, count in class_counts.items()}
    scale_pos_weight = class_counts[0] / class_counts[1]
    print("   Class weights: " + str(class_weights))
    print("   XGBoost scale_pos_weight: " + str(round(scale_pos_weight, 2)))
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "class_weights": class_weights,
        "scale_pos_weight": scale_pos_weight,
        "original_X_train": X_train,
        "original_X_test": X_test
    }

if __name__ == "__main__":
    results = run_preprocessing()
