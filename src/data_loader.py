"""Dataset loading utilities for the Customer Churn Prediction project."""
from pathlib import Path
from typing import Optional
import pandas as pd

from .config import RAW_DATA_PATH, ID_COLUMN, TARGET_COLUMN


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV file. Defaults to RAW_DATA_PATH from config.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all columns as loaded from CSV.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist at the specified path.
    """
    data_path = path or RAW_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Please download the IBM Telco Customer Churn dataset from "
            f"https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            f"and place it in data/raw/"
        )
    df = pd.read_csv(data_path)
    return df


def get_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and target.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Features dataframe (X) and target series (y).

    Notes
    -----
    - Drops the customerID column as it is an identifier, not a feature.
    - Target column is specified by TARGET_COLUMN in config.
    """
    # Drop identifier column
    df_features = df.drop(columns=[ID_COLUMN], errors="ignore")
    # Separate target
    y = df_features[TARGET_COLUMN]
    X = df_features.drop(columns=[TARGET_COLUMN])
    return X, y


def get_data_summary(df: pd.DataFrame) -> dict:
    """Generate a summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    dict
        Dictionary containing shape, columns, dtypes, missing values, and duplicates.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }
    return summary


def print_data_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary of the dataset."""
    summary = get_data_summary(df)
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {summary['shape'][0]} rows x {summary['shape'][1]} columns")
    print(f"Memory usage: {summary['memory_usage_mb']} MB")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print()
    print("Columns and Data Types:")
    for col, dtype in summary["dtypes"].items():
        missing = summary["missing_values"][col]
        print(f"  {col:<25} {dtype:<15} Missing: {missing}")
    print("=" * 60)
