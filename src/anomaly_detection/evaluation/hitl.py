"""
evaluation.hitl
---------------
Human-in-the-loop (HITL) precision simulation.

Simulates the weekly control-team review loop: a random sample of flagged
records is "reviewed" each week and labeled as true anomaly / false alarm.
Precision accumulates over time, mirroring how trust in the model builds
in a real governance workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HITLResult:
    weekly: pd.DataFrame     # columns: week, cumulative_precision, tp, fp
    final_precision: float
    total_reviewed: int
    total_tp: int
    total_fp: int


def simulate_hitl_review(
    flagged_indices: np.ndarray,
    labels_true: np.ndarray,
    weekly_sample: int = 20,
    n_weeks: int = 12,
    random_state: int = 42,
) -> HITLResult:
    """Simulate a multi-week HITL review of flagged anomalies.

    Each week a fresh sample of un-reviewed flagged records is drawn,
    "reviewed" (ground-truth label looked up), and added to the cumulative
    precision tally.

    Parameters
    ----------
    flagged_indices:
        Row indices of customers the model flagged as anomalous.
    labels_true:
        Full ground-truth label array (0 = normal, 1 = anomaly).
    weekly_sample:
        Maximum number of records reviewed per week.
    n_weeks:
        Number of weeks to simulate.
    random_state:
        NumPy seed.

    Returns
    -------
    :class:`HITLResult`
    """
    rng = np.random.default_rng(random_state)
    reviewed: set[int] = set()
    cum_tp = cum_fp = 0
    rows = []

    for week in range(1, n_weeks + 1):
        pool = [i for i in flagged_indices if i not in reviewed]
        if not pool:
            break
        batch = rng.choice(pool, size=min(weekly_sample, len(pool)), replace=False)
        reviewed.update(batch.tolist())

        tp = int(labels_true[batch].sum())
        fp = len(batch) - tp
        cum_tp += tp
        cum_fp += fp

        prec = cum_tp / (cum_tp + cum_fp + 1e-9)
        rows.append({"week": week, "cumulative_precision": prec, "tp": cum_tp, "fp": cum_fp})

    df = pd.DataFrame(rows)
    return HITLResult(
        weekly=df,
        final_precision=float(df["cumulative_precision"].iloc[-1]) if len(df) else 0.0,
        total_reviewed=len(reviewed),
        total_tp=cum_tp,
        total_fp=cum_fp,
    )
