"""Tests for evaluation modules: synthetic, hitl, psi."""

import numpy as np
import pandas as pd
import pytest

from anomaly_detection.evaluation.synthetic import inject_anomalies, FEATURE_NAMES
from anomaly_detection.evaluation.hitl import simulate_hitl_review
from anomaly_detection.evaluation.psi import compute_psi, monitor_psi


# ── Synthetic injection ───────────────────────────────────────────────────────

def test_inject_anomalies_label_count(feature_matrix):
    result = inject_anomalies(feature_matrix, frac=0.05, random_state=0)
    assert result.labels_true.sum() == result.n_injected


def test_inject_anomalies_shape_unchanged(feature_matrix):
    result = inject_anomalies(feature_matrix, frac=0.05, random_state=0)
    assert result.X_injected.shape == feature_matrix.shape


def test_inject_anomalies_values_differ(feature_matrix):
    result = inject_anomalies(feature_matrix, frac=0.05, random_state=0)
    # At least some rows must be modified
    changed = np.any(result.X_injected != feature_matrix, axis=1)
    assert changed[result.synth_indices].any()


def test_inject_anomalies_fraction(feature_matrix):
    result = inject_anomalies(feature_matrix, frac=0.10, random_state=0)
    expected = int(len(feature_matrix) * 0.10)
    assert result.n_injected == expected


# ── HITL simulation ───────────────────────────────────────────────────────────

def test_hitl_runs_n_weeks():
    flagged = np.arange(100)
    labels  = np.zeros(200, dtype=int)
    labels[:20] = 1
    result = simulate_hitl_review(flagged, labels, weekly_sample=15, n_weeks=5)
    assert len(result.weekly) == 5


def test_hitl_precision_in_range():
    flagged = np.arange(50)
    labels  = np.zeros(100, dtype=int)
    labels[:10] = 1
    result = simulate_hitl_review(flagged, labels, weekly_sample=10, n_weeks=6)
    assert 0.0 <= result.final_precision <= 1.0


def test_hitl_reviewed_count():
    flagged = np.arange(60)
    labels  = np.zeros(100, dtype=int)
    result = simulate_hitl_review(flagged, labels, weekly_sample=10, n_weeks=4)
    assert result.total_reviewed == min(40, len(flagged))


# ── PSI ───────────────────────────────────────────────────────────────────────

def test_compute_psi_identical_distributions():
    x = np.random.default_rng(0).normal(0, 1, 500)
    psi = compute_psi(x, x)
    assert psi < 0.01


def test_compute_psi_different_distributions():
    rng = np.random.default_rng(0)
    baseline   = rng.normal(0, 1, 500)
    monitoring = rng.normal(5, 1, 500)   # clear shift
    psi = compute_psi(baseline, monitoring)
    assert psi > 0.2


def test_monitor_psi_returns_all_features(feature_matrix):
    df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    half = len(df) // 2
    result = monitor_psi(df.iloc[:half], df.iloc[half:], FEATURE_NAMES)
    assert set(result["feature"]) == set(FEATURE_NAMES)


def test_monitor_psi_status_values(feature_matrix):
    df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    half = len(df) // 2
    result = monitor_psi(df.iloc[:half], df.iloc[half:], FEATURE_NAMES)
    valid = {"STABLE", "SLIGHT SHIFT", "SIGNIFICANT SHIFT"}
    assert set(result["status"]).issubset(valid)
