import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_telco_df() -> pd.DataFrame:
    rows = 60
    data = {
        "customerID": [f"C{i:04d}" for i in range(rows)],
        "gender": ["Female", "Male"] * (rows // 2),
        "SeniorCitizen": [0, 1] * (rows // 2),
        "Partner": ["Yes", "No", "No"] * (rows // 3),
        "Dependents": ["No", "Yes"] * (rows // 2),
        "tenure": [0 if i == 0 else (i % 60) + 1 for i in range(rows)],
        "PhoneService": ["Yes", "No"] * (rows // 2),
        "MultipleLines": ["No", "Yes", "No phone service"] * (rows // 3),
        "InternetService": ["DSL", "Fiber optic", "No"] * (rows // 3),
        "OnlineSecurity": ["No", "Yes", "No internet service"] * (rows // 3),
        "OnlineBackup": ["Yes", "No", "No internet service"] * (rows // 3),
        "DeviceProtection": ["No", "Yes", "No internet service"] * (rows // 3),
        "TechSupport": ["No", "Yes", "No internet service"] * (rows // 3),
        "StreamingTV": ["No", "Yes", "No internet service"] * (rows // 3),
        "StreamingMovies": ["Yes", "No", "No internet service"] * (rows // 3),
        "Contract": ["Month-to-month", "One year", "Two year"] * (rows // 3),
        "PaperlessBilling": ["Yes", "No"] * (rows // 2),
        "PaymentMethod": [
            "Electronic check", "Mailed check", "Bank transfer (automatic)"
        ] * (rows // 3),
        "MonthlyCharges": np.linspace(20.0, 110.0, rows),
        "TotalCharges": ["" if i in (0, 1) else str(float(i * 10 + 20)) for i in range(rows)],
        "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(rows)],
    }
    return pd.DataFrame(data)
