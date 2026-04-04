"""Tests for features.engineer."""

import numpy as np
import pytest

from anomaly_detection.features.engineer import (
    FEATURE_NAMES,
    build_customer_features,
    BehavioralFeatureTransformer,
)


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 11


def test_build_customer_features_shape(sample_transactions):
    feat = build_customer_features(sample_transactions)
    n_customers = sample_transactions["Customer ID"].nunique()
    assert feat.shape == (n_customers, len(FEATURE_NAMES))


def test_build_customer_features_no_nulls(sample_transactions):
    feat = build_customer_features(sample_transactions)
    assert feat[FEATURE_NAMES].isna().sum().sum() == 0


def test_build_customer_features_non_negative_counts(sample_transactions):
    feat = build_customer_features(sample_transactions)
    assert (feat["total_orders"] >= 1).all()
    assert (feat["total_revenue"] > 0).all()
    assert (feat["recency_days"] >= 0).all()


def test_transformer_fit_transform_shape(sample_transactions):
    t = BehavioralFeatureTransformer()
    X, ids = t.fit_transform(sample_transactions)
    n_customers = sample_transactions["Customer ID"].nunique()
    assert X.shape == (n_customers, len(FEATURE_NAMES))
    assert len(ids) == n_customers


def test_transformer_scaled_no_nans(sample_transactions):
    t = BehavioralFeatureTransformer()
    X, _ = t.fit_transform(sample_transactions)
    assert not np.isnan(X).any()
