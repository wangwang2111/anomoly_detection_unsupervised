"""Shared fixtures for all test modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_transactions() -> pd.DataFrame:
    """Minimal synthetic transaction DataFrame that passes loader.clean() expectations."""
    rng = np.random.default_rng(0)
    n = 500
    customers = [str(c) for c in rng.integers(1000, 2000, size=n)]
    invoices  = [f"INV{i:05d}" for i in range(n)]
    dates     = pd.date_range("2020-01-01", periods=n, freq="6h")
    qty   = rng.integers(1, 50, size=n)
    price = rng.uniform(1.0, 100.0, size=n)
    return pd.DataFrame({
        "Invoice":     invoices,
        "StockCode":   rng.choice(["A1", "B2", "C3", "D4"], size=n),
        "Description": "item",
        "Quantity":    qty,
        "InvoiceDate": dates,
        "Price":       price,
        "Customer ID": customers,
        "Country":     "UK",
        "Revenue":     qty * price,   # added by loader.clean() in production
    })


@pytest.fixture(scope="session")
def feature_matrix(sample_transactions) -> np.ndarray:
    """Raw feature matrix from the sample transactions."""
    from anomaly_detection.features.engineer import build_customer_features, FEATURE_NAMES
    feat = build_customer_features(sample_transactions)
    return feat[FEATURE_NAMES].values
