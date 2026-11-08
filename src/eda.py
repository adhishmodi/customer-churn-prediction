"""Exploratory Data Analysis for Customer Churn Prediction."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import REPORTS_DIR, TARGET_COLUMN, ID_COLUMN
from .data_loader import load_raw_data, get_data_summary

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

FIGURES_DIR = REPORTS_DIR / "figures"


def load_and_prepare_data() -> pd.DataFrame:
    """Load data and apply only deterministic EDA conversions."""
    df = load_raw_data()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    zero_tenure_missing = df["TotalCharges"].isna() & df["tenure"].eq(0)
    df.loc[zero_tenure_missing, "TotalCharges"] = 0.0
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})
    df["ChurnBinary"] = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    return df


def save_figure(fig, name: str, dpi: int = 150) -> Path:
    """Save a generated figure under reports/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


