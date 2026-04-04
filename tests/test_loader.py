"""Tests for data.loader."""

import pandas as pd
import pytest

from anomaly_detection.data.loader import clean


def _make_raw(n: int = 20) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "Invoice":     [f"INV{i}" for i in range(n)],
        "StockCode":   ["A"] * n,
        "Description": ["item"] * n,
        "Quantity":    rng.integers(1, 10, size=n),
        "InvoiceDate": pd.date_range("2021-01-01", periods=n, freq="D"),
        "Price":       rng.uniform(1, 50, size=n),
        "Customer ID": [float(i % 5 + 100) for i in range(n)],
        "Country":     ["UK"] * n,
    })


def test_clean_removes_missing_customer_id():
    df = _make_raw()
    df.loc[0, "Customer ID"] = float("nan")
    result = clean(df)
    assert result["Customer ID"].notna().all()


def test_clean_removes_cancellations():
    df = _make_raw()
    df.loc[1, "Invoice"] = "CINV1"
    result = clean(df)
    assert not result["Invoice"].astype(str).str.startswith("C").any()


def test_clean_removes_non_positive_quantity():
    df = _make_raw()
    df.loc[2, "Quantity"] = -5
    df.loc[3, "Quantity"] = 0
    result = clean(df)
    assert (result["Quantity"] > 0).all()


def test_clean_adds_revenue_column():
    df = _make_raw()
    result = clean(df)
    assert "Revenue" in result.columns
    assert (result["Revenue"] > 0).all()


def test_clean_customer_id_is_string():
    df = _make_raw()
    result = clean(df)
    assert result["Customer ID"].dtype == object


def test_clean_drops_duplicates():
    df = _make_raw(10)
    df_dup = pd.concat([df, df.iloc[:3]], ignore_index=True)
    result = clean(df_dup)
    assert len(result) == 10
