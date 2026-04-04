"""
evaluation.psi
--------------
Population Stability Index (PSI) for feature drift monitoring.

PSI measures how much a feature's distribution has shifted between a
baseline period and a monitoring period.

  PSI < 0.10  →  stable
  0.10–0.20   →  slight shift (investigate)
  > 0.20      →  significant shift (retrain / alert)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_STABLE      = "STABLE"
_SLIGHT      = "SLIGHT SHIFT"
_SIGNIFICANT = "SIGNIFICANT SHIFT"


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute PSI between two 1-D arrays.

    Bin boundaries are derived from the ``expected`` distribution so the
    same bins are applied to ``actual``.

    Parameters
    ----------
    expected:
        Baseline distribution (e.g. year 1 customer features).
    actual:
        Monitoring distribution (e.g. year 2 customer features).
    n_bins:
        Number of percentile-based bins.

    Returns
    -------
    Scalar PSI value.
    """
    breakpoints = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(expected, breakpoints))

    # If all values are identical, distribution is trivially stable
    if len(bin_edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual,   bins=bin_edges)

    exp_pct = np.clip(exp_counts / len(expected), 1e-4, None)
    act_pct = np.clip(act_counts / len(actual),   1e-4, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def monitor_psi(
    baseline: pd.DataFrame,
    monitoring: pd.DataFrame,
    features: list[str],
    n_bins: int = 10,
    slight_threshold: float = 0.10,
    significant_threshold: float = 0.20,
) -> pd.DataFrame:
    """Compute PSI for each feature and return a summary DataFrame.

    Parameters
    ----------
    baseline:
        DataFrame for the baseline period (must contain ``features`` columns).
    monitoring:
        DataFrame for the monitoring period (must contain ``features`` columns).
    features:
        List of feature column names to monitor.
    n_bins:
        Passed to :func:`compute_psi`.
    slight_threshold:
        PSI value above which a feature is flagged as "SLIGHT SHIFT".
    significant_threshold:
        PSI value above which a feature is flagged as "SIGNIFICANT SHIFT".

    Returns
    -------
    DataFrame with columns: feature, psi, status — sorted by psi descending.
    """
    rows = []
    for feat in features:
        psi_val = compute_psi(
            baseline[feat].dropna().values,
            monitoring[feat].dropna().values,
            n_bins=n_bins,
        )
        if psi_val >= significant_threshold:
            status = _SIGNIFICANT
        elif psi_val >= slight_threshold:
            status = _SLIGHT
        else:
            status = _STABLE

        rows.append({"feature": feat, "psi": round(psi_val, 4), "status": status})

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
