"""Tests for models.detector."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from anomaly_detection.models.detector import AnomalyDetectorSuite, DetectionResult


@pytest.fixture
def small_X():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 11))
    scaler = StandardScaler()
    return scaler.fit_transform(X)


@pytest.fixture
def labels(small_X):
    labels = np.zeros(len(small_X), dtype=int)
    labels[:10] = 1
    return labels


@pytest.fixture
def cfg():
    return {
        "contamination": 0.05,
        "random_state": 42,
        "isolation_forest": {"n_estimators": 50},
        "lof": {"n_neighbors": 10},
        "hbos": {"n_bins": 10},
    }


def test_fit_predict_returns_detection_result(small_X, labels, cfg):
    suite = AnomalyDetectorSuite(cfg, mlflow_cfg=None)
    result = suite.fit_predict(small_X, labels)
    assert isinstance(result, DetectionResult)


def test_all_three_models_present(small_X, labels, cfg):
    suite = AnomalyDetectorSuite(cfg, mlflow_cfg=None)
    result = suite.fit_predict(small_X, labels)
    assert set(result.models.keys()) == {"IsolationForest", "LOF", "HBOS"}


def test_labels_binary(small_X, labels, cfg):
    suite = AnomalyDetectorSuite(cfg, mlflow_cfg=None)
    result = suite.fit_predict(small_X, labels)
    for r in result.models.values():
        assert set(np.unique(r.labels)).issubset({0, 1})
    assert set(np.unique(result.ensemble_labels)).issubset({0, 1})


def test_scores_finite(small_X, labels, cfg):
    suite = AnomalyDetectorSuite(cfg, mlflow_cfg=None)
    result = suite.fit_predict(small_X, labels)
    for r in result.models.values():
        assert np.isfinite(r.scores).all()
    assert np.isfinite(result.ensemble_scores).all()


def test_ensemble_scores_normalised(small_X, labels, cfg):
    suite = AnomalyDetectorSuite(cfg, mlflow_cfg=None)
    result = suite.fit_predict(small_X, labels)
    # Ensemble scores are min-max normalised averages → in [0, 1]
    assert result.ensemble_scores.min() >= -1e-9
    assert result.ensemble_scores.max() <= 1 + 1e-9


def test_best_model_name_valid(small_X, labels, cfg):
    suite = AnomalyDetectorSuite(cfg, mlflow_cfg=None)
    result = suite.fit_predict(small_X, labels)
    assert result.best_model_name in result.models


def test_contamination_flag_count(small_X, labels, cfg):
    suite = AnomalyDetectorSuite(cfg, mlflow_cfg=None)
    result = suite.fit_predict(small_X, labels)
    expected = int(len(small_X) * cfg["contamination"])
    for r in result.models.values():
        # PyOD flags exactly contamination * n rows
        assert r.labels.sum() == expected
