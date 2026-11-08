"""Data validation checks for the Customer Churn Prediction project."""
from typing import Any
import pandas as pd
import numpy as np

from .config import TARGET_COLUMN, ID_COLUMN


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Run comprehensive validation checks on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Returns
    -------
    dict
        Validation results including any warnings or errors.

    Raises
    ------
    DataValidationError
        If critical validation checks fail.
    """
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }

    # 1. Check required columns exist
    required_columns = [ID_COLUMN, TARGET_COLUMN]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        results["errors"].append(f"Missing required columns: {missing_cols}")
        results["passed"] = False

    # 2. Check target column values
    if TARGET_COLUMN in df.columns:
        target_values = df[TARGET_COLUMN].unique()
        expected_values = {"Yes", "No"}
        unexpected = set(target_values) - expected_values
        if unexpected:
            results["warnings"].append(
                f"Target column has unexpected values: {unexpected}. Expected: {expected_values}"
            )
        # Check class balance
        class_counts = df[TARGET_COLUMN].value_counts()
        results["info"]["class_distribution"] = class_counts.to_dict()
        if len(class_counts) == 2:
            minority_pct = class_counts.min() / class_counts.sum() * 100
            results["info"]["minority_class_pct"] = round(minority_pct, 2)
            if minority_pct < 10:
                results["warnings"].append(
                    f"Severe class imbalance: minority class is only {minority_pct:.1f}%"
                )
            elif minority_pct < 20:
                results["warnings"].append(
                    f"Moderate class imbalance: minority class is {minority_pct:.1f}%"
                )

    # 3. Check for missing values
    missing = df.isna().sum()
    cols_with_missing = missing[missing > 0]
    if not cols_with_missing.empty:
        results["warnings"].append(
            f"Columns with missing values: {cols_with_missing.to_dict()}"
        )
        results["info"]["missing_values"] = cols_with_missing.to_dict()

    # 4. Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        results["warnings"].append(f"Found {dup_count} duplicate rows")
        results["info"]["duplicate_count"] = int(dup_count)

    # 5. Check identifier column uniqueness
    if ID_COLUMN in df.columns:
        id_unique = df[ID_COLUMN].nunique()
        total_rows = len(df)
        if id_unique != total_rows:
            results["errors"].append(
                f"ID column '{ID_COLUMN}' has {total_rows - id_unique} duplicate values"
            )
            results["passed"] = False
        results["info"]["unique_ids"] = id_unique

    # 6. Check TotalCharges column (known issue: blank strings)
    if "TotalCharges" in df.columns:
        # Check if it's object type (strings) - needs conversion
        if df["TotalCharges"].dtype == "object":
            non_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum()
            if non_numeric > 0:
                results["warnings"].append(
                    f"TotalCharges has {non_numeric} non-numeric values (likely blank strings). "
                    f"Conversion to numeric required."
                )
                results["info"]["totalcharges_non_numeric"] = int(non_numeric)

    # 7. Check numeric columns for reasonable ranges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != ID_COLUMN:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < 0 and col not in ["tenure"]:  # tenure can be 0
                results["warnings"].append(
                    f"Column '{col}' has negative values (min={min_val})"
                )
            results["info"][f"{col}_range"] = {"min": float(min_val), "max": float(max_val)}

    # 8. Check categorical columns cardinality
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col != ID_COLUMN:
            n_unique = df[col].nunique()
            if n_unique > 50:
                results["warnings"].append(
                    f"High cardinality categorical column '{col}': {n_unique} unique values"
                )
            results["info"][f"{col}_cardinality"] = n_unique

    if results["errors"]:
        raise DataValidationError(f"Validation failed: {results['errors']}")

    return results


def print_validation_results(results: dict[str, Any]) -> None:
    """Print formatted validation results."""
    print("=" * 60)
    print("DATA VALIDATION RESULTS")
    print("=" * 60)
    print(f"Overall: {'PASSED' if results['passed'] else 'FAILED'}")
    print()

    if results["errors"]:
        print("ERRORS:")
        for err in results["errors"]:
            print(f"  [ERROR] {err}")
        print()

    if results["warnings"]:
        print("WARNINGS:")
        for warn in results["warnings"]:
            print(f"  [WARN] {warn}")
        print()

    if results["info"]:
        print("INFO:")
        for key, value in results["info"].items():
            print(f"  [INFO] {key}: {value}")
        print()

    print("=" * 60)
