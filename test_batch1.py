"""Test script to verify Batch 1 environment and dataset loading."""
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        import seaborn
        import sklearn
        import xgboost
        import shap
        import streamlit
        import pytest
        print("  [OK] All core packages imported successfully")
        print(f"     pandas: {pd.__version__}")
        print(f"     numpy: {np.__version__}")
        print(f"     scikit-learn: {sklearn.__version__}")
        print(f"     xgboost: {xgboost.__version__}")
        print(f"     shap: {shap.__version__}")
        print(f"     streamlit: {streamlit.__version__}")
        print(f"     pytest: {pytest.__version__}")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_src_module():
    """Test that src module can be imported."""
    print("\nTesting src module...")
    try:
        from src import (
            load_raw_data,
            get_data_summary,
            print_data_summary,
            validate_dataset,
            print_validation_results,
            RAW_DATA_PATH,
            RANDOM_SEED,
            TARGET_COLUMN,
        )
        print("  [OK] src module imported successfully")
        print(f"     RAW_DATA_PATH: {RAW_DATA_PATH}")
        print(f"     RANDOM_SEED: {RANDOM_SEED}")
        print(f"     TARGET_COLUMN: {TARGET_COLUMN}")
        return True
    except ImportError as e:
        print(f"  [FAIL] src module import failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """Test loading the dataset."""
    print("\nTesting data loading...")
    try:
        from src import load_raw_data, print_data_summary
        df = load_raw_data()
        print(f"  [OK] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        print_data_summary(df)
        return df
    except Exception as e:
        print(f"  [FAIL] Data loading failed: {e}")
        traceback.print_exc()
        return None


def test_validation(df):
    """Test data validation."""
    print("\nTesting data validation...")
    try:
        from src import validate_dataset, print_validation_results
        results = validate_dataset(df)
        print_validation_results(results)
        return results
    except Exception as e:
        print(f"  [FAIL] Validation failed: {e}")
        traceback.print_exc()
        return None


def main():
    """Run all Batch 1 tests."""
    print("=" * 60)
    print("BATCH 1 VERIFICATION TESTS")
    print("=" * 60)

    all_passed = True

    # Test 1: Imports
    if not test_imports():
        all_passed = False

    # Test 2: src module
    if not test_src_module():
        all_passed = False

    # Test 3: Data loading
    df = test_data_loading()
    if df is None:
        all_passed = False

    # Test 4: Validation
    if df is not None:
        results = test_validation(df)
        if results is None or not results.get("passed", False):
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL BATCH 1 TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("[FAILURE] SOME BATCH 1 TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
