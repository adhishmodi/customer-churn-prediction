from pathlib import Path

import pytest

from src.data_loader import load_raw_data


def test_load_raw_data_reports_missing_dataset(tmp_path):
    missing_path = tmp_path / "not-present.csv"
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_raw_data(missing_path)


def test_load_raw_data_reads_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("customerID,Churn\nA1,No\nA2,Yes\n", encoding="utf-8")
    frame = load_raw_data(csv_path)
    assert frame.shape == (2, 2)
