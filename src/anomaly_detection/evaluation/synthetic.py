"""
evaluation.synthetic
--------------------
Synthetic anomaly injection for recall estimation.

Since the dataset has no ground-truth labels, we inject *known* behavioral
outliers into the scaled feature matrix and measure whether each detector
flags them.  The injection types mirror real-world B2B anomaly patterns.

Injection types
~~~~~~~~~~~~~~~
volume_spike       : order quantity and revenue multiplied by 3–5×
revenue_collapse   : revenue and quantity reduced to 10% of baseline
frequency_drop     : recency_days inflated 4–6× (customer went silent)
timing_irregular   : inter-order std inflated 5–10× (chaotic timing)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anomaly_detection.features.engineer import FEATURE_NAMES


_INJECTION_MAP: dict[str, dict] = {
    "volume_spike":      {"features": ["mean_order_qty", "total_revenue"], "scale": (3.0, 5.0)},
    "revenue_collapse":  {"features": ["total_revenue", "mean_order_qty"], "scale": (0.08, 0.12)},
    "frequency_drop":    {"features": ["recency_days"],                    "scale": (4.0, 6.0)},
    "timing_irregular":  {"features": ["iod_std"],                         "scale": (5.0, 10.0)},
}


@dataclass
class InjectionResult:
    X_injected: np.ndarray        # raw (unscaled) feature matrix with injections
    labels_true: np.ndarray       # 0 = normal, 1 = injected anomaly
    synth_indices: np.ndarray     # row indices of injected customers
    injection_types: np.ndarray   # type string per injected customer
    n_injected: int


def inject_anomalies(
    X_raw: np.ndarray,
    frac: float = 0.05,
    injection_types: list[str] | None = None,
    random_state: int = 42,
) -> InjectionResult:
    """Inject synthetic anomalies into the raw (unscaled) feature matrix.

    Parameters
    ----------
    X_raw:
        Unscaled customer feature matrix (n_customers × n_features).
        Column order must match :data:`~anomaly_detection.features.engineer.FEATURE_NAMES`.
    frac:
        Fraction of customers to inject as anomalies.
    injection_types:
        Subset of injection types to use.  Defaults to all four.
    random_state:
        NumPy seed for reproducibility.

    Returns
    -------
    :class:`InjectionResult`
    """
    rng = np.random.default_rng(random_state)
    feat_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    if injection_types is None:
        injection_types = list(_INJECTION_MAP.keys())

    n = len(X_raw)
    n_synth = max(1, int(n * frac))
    synth_idx = rng.choice(n, size=n_synth, replace=False)

    X_out = X_raw.copy().astype(float)
    labels = np.zeros(n, dtype=int)
    labels[synth_idx] = 1

    types_assigned = rng.choice(injection_types, size=n_synth)

    for row_i, itype in zip(synth_idx, types_assigned):
        spec = _INJECTION_MAP[itype]
        lo, hi = spec["scale"]
        multiplier = rng.uniform(lo, hi)
        for feat_name in spec["features"]:
            col = feat_idx[feat_name]
            X_out[row_i, col] = X_out[row_i, col] * multiplier

    return InjectionResult(
        X_injected=X_out,
        labels_true=labels,
        synth_indices=synth_idx,
        injection_types=types_assigned,
        n_injected=n_synth,
    )
