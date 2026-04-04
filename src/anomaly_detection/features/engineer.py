"""
features.engineer
-----------------
Transforms a clean transaction DataFrame into a customer-level feature
matrix.  Uses a sklearn-compatible transformer pattern so the same
object can fit on training data and transform held-out periods.

Feature groups
~~~~~~~~~~~~~~
- RFM          : recency_days, total_orders, total_revenue
- Order value  : mean_order_value, cv_order_value
- Order volume : mean_order_qty, cv_order_qty
- Basket       : mean_basket_size (unique SKUs per order)
- Timing       : iod_mean, iod_std  (inter-order days)
- Trend        : revenue_slope (linear slope of monthly revenue)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES: list[str] = [
    "total_orders",
    "total_revenue",
    "mean_order_value",
    "cv_order_value",
    "mean_order_qty",
    "cv_order_qty",
    "mean_basket_size",
    "recency_days",
    "iod_mean",
    "iod_std",
    "revenue_slope",
]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _inter_order_stats(dates: pd.Series) -> pd.Series:
    """Return mean and std of gaps (in days) between consecutive orders."""
    sorted_dates = sorted(dates)
    if len(sorted_dates) < 2:
        return pd.Series({"iod_mean": np.nan, "iod_std": np.nan})
    gaps = [(sorted_dates[i + 1] - sorted_dates[i]).days for i in range(len(sorted_dates) - 1)]
    return pd.Series({"iod_mean": float(np.mean(gaps)), "iod_std": float(np.std(gaps))})


def _revenue_slope(grp: pd.DataFrame, min_months: int = 3) -> float:
    """OLS slope of monthly revenue over time. Returns NaN for short histories."""
    if len(grp) < min_months:
        return np.nan
    x = grp["month_idx"].values.astype(float)
    y = grp["Revenue"].values.astype(float)
    return float(np.polyfit(x, y, 1)[0])


# ── Public API ────────────────────────────────────────────────────────────────

def build_customer_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
    min_slope_months: int = 3,
) -> pd.DataFrame:
    """Aggregate transactions into a per-customer behavioral feature matrix.

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame (output of :func:`data.loader.clean`).
    snapshot_date:
        Reference date for recency calculation.  Defaults to max InvoiceDate + 1 day.
    min_slope_months:
        Minimum number of active months required to compute revenue_slope.

    Returns
    -------
    DataFrame indexed by Customer ID with one row per customer and columns
    matching :data:`FEATURE_NAMES`.
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # ── Invoice-level aggregation ─────────────────────────────────────────────
    inv = (
        df.groupby(["Customer ID", "Invoice", "InvoiceDate"])
        .agg(
            invoice_revenue=("Revenue", "sum"),
            invoice_qty=("Quantity", "sum"),
            unique_skus=("StockCode", "nunique"),
        )
        .reset_index()
    )

    # ── Inter-order timing ────────────────────────────────────────────────────
    iod = (
        inv.groupby("Customer ID")["InvoiceDate"]
        .apply(_inter_order_stats)
        .unstack()
        .reset_index()
    )

    # ── Customer-level aggregation ────────────────────────────────────────────
    cust = (
        inv.groupby("Customer ID")
        .agg(
            total_orders=("Invoice", "nunique"),
            total_revenue=("invoice_revenue", "sum"),
            mean_order_value=("invoice_revenue", "mean"),
            std_order_value=("invoice_revenue", "std"),
            mean_order_qty=("invoice_qty", "mean"),
            std_order_qty=("invoice_qty", "std"),
            mean_basket_size=("unique_skus", "mean"),
            last_order_date=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    cust["recency_days"] = (snapshot_date - cust["last_order_date"]).dt.days
    cust = cust.merge(iod, on="Customer ID", how="left")

    # CV: 0 for single-order customers (no variability observed)
    cust["cv_order_value"] = cust["std_order_value"].fillna(0) / (cust["mean_order_value"] + 1e-9)
    cust["cv_order_qty"] = cust["std_order_qty"].fillna(0) / (cust["mean_order_qty"] + 1e-9)

    # ── Revenue trend ─────────────────────────────────────────────────────────
    df = df.copy()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    monthly = df.groupby(["Customer ID", "YearMonth"])["Revenue"].sum().reset_index()
    monthly["month_idx"] = monthly.groupby("Customer ID")["YearMonth"].transform(
        lambda s: (s - s.min()).apply(lambda p: p.n)
    )

    slopes = (
        monthly.groupby("Customer ID")
        .apply(lambda g: _revenue_slope(g, min_slope_months), include_groups=False)
        .rename("revenue_slope")
        .reset_index()
    )
    cust = cust.merge(slopes, on="Customer ID", how="left")

    # ── Imputation ────────────────────────────────────────────────────────────
    feat = cust[["Customer ID"] + FEATURE_NAMES].copy()
    feat["iod_mean"] = feat["iod_mean"].fillna(feat["iod_mean"].median())
    feat["iod_std"] = feat["iod_std"].fillna(0.0)
    feat["revenue_slope"] = feat["revenue_slope"].fillna(0.0)
    for col in FEATURE_NAMES:
        if feat[col].isna().any():
            feat[col] = feat[col].fillna(feat[col].median())

    return feat.set_index("Customer ID")


class BehavioralFeatureTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper around :func:`build_customer_features`.

    Adds StandardScaler + median SimpleImputer so the object can be
    persisted and reused on new data without re-fitting the scaler.

    Parameters
    ----------
    min_slope_months:
        Passed through to :func:`build_customer_features`.
    """

    def __init__(self, min_slope_months: int = 3) -> None:
        self.min_slope_months = min_slope_months
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, y=None) -> "BehavioralFeatureTransformer":
        feat = build_customer_features(df, min_slope_months=self.min_slope_months)
        X = feat[FEATURE_NAMES].values
        X = self._imputer.fit_transform(X)
        self._scaler.fit(X)
        self._snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        self.feature_names_out_ = FEATURE_NAMES
        self.customer_ids_ = feat.index.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Return (X_scaled, customer_ids)."""
        feat = build_customer_features(
            df,
            snapshot_date=self._snapshot_date,
            min_slope_months=self.min_slope_months,
        )
        X = feat[FEATURE_NAMES].values
        X = self._imputer.transform(X)
        X = self._scaler.transform(X)
        return X, feat.index.tolist()

    def fit_transform(self, df: pd.DataFrame, y=None) -> tuple[np.ndarray, list[str]]:
        self.fit(df)
        feat = build_customer_features(
            df,
            snapshot_date=self._snapshot_date,
            min_slope_months=self.min_slope_months,
        )
        X = feat[FEATURE_NAMES].values
        X = self._imputer.transform(X)
        X_scaled = self._scaler.transform(X)
        self.customer_ids_ = feat.index.tolist()
        return X_scaled, feat.index.tolist()
