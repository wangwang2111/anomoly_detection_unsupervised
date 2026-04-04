"""Script to write the thin orchestration notebook from scratch."""
import json, uuid
from pathlib import Path

def cell(source: str, cell_type: str = "code") -> dict:
    return {
        "cell_type": cell_type,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": source,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }

cells = []

cells.append(cell("""# B2B Customer Behavioral Anomaly Detection
Full pipeline via modular `src/anomaly_detection/` package.
All logic lives in the modules — this notebook is the orchestration layer.

---
## 0. Setup""", "markdown"))

cells.append(cell("""\
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# Ensure src/ is on path when running without pip install
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless — remove if running interactively

from anomaly_detection import load_config
from anomaly_detection.data.loader import load_raw, clean
from anomaly_detection.features.engineer import (
    FEATURE_NAMES, build_customer_features, BehavioralFeatureTransformer,
)
from anomaly_detection.models.detector import AnomalyDetectorSuite
from anomaly_detection.evaluation.synthetic import inject_anomalies
from anomaly_detection.evaluation.hitl import simulate_hitl_review
from anomaly_detection.evaluation.psi import monitor_psi
from anomaly_detection.visualization import plots
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

cfg = load_config(ROOT / "config" / "config.yaml")
print("Config loaded:", list(cfg.keys()))
"""))

cells.append(cell("---\n## 1. Data Loading & Cleaning", "markdown"))

cells.append(cell("""\
df_raw = load_raw(ROOT / cfg["data"]["path"], sheets=cfg["data"]["sheets"])
print(f"Raw shape: {df_raw.shape}")

df = clean(df_raw)
print(f"Cleaned: {df.shape[0]:,} rows | {df['Customer ID'].nunique():,} customers")
print(f"Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
df.head(3)
"""))

cells.append(cell("---\n## 2. Feature Engineering\n11 behavioral features per customer: RFM + variability + timing + revenue trend.", "markdown"))

cells.append(cell("""\
feat_df = build_customer_features(
    df, min_slope_months=cfg["features"]["min_slope_months"]
)
print(f"Feature matrix: {feat_df.shape}  |  NaNs: {feat_df[FEATURE_NAMES].isna().sum().sum()}")
feat_df[FEATURE_NAMES].describe().round(2)
"""))

cells.append(cell("""\
fig = plots.plot_correlation_heatmap(feat_df.reset_index(), FEATURE_NAMES)
plt.show()
"""))

cells.append(cell("---\n## 3. Synthetic Anomaly Injection\nInject known outliers → measure **recall** (can the model find what we planted?).", "markdown"))

cells.append(cell("""\
eval_cfg = cfg["evaluation"]
X_raw = feat_df[FEATURE_NAMES].values

injection = inject_anomalies(
    X_raw,
    frac=eval_cfg["synthetic_frac"],
    injection_types=eval_cfg["injection_types"],
    random_state=cfg["models"]["random_state"],
)

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
X_inj_scaled = scaler.fit_transform(imputer.fit_transform(injection.X_injected))

import pandas as pd
print(f"Injected {injection.n_injected} anomalies ({eval_cfg['synthetic_frac']:.0%} of {len(X_raw)})")
pd.Series(injection.injection_types).value_counts().rename("count").to_frame()
"""))

cells.append(cell("---\n## 4. Model Training & Evaluation\nIsolation Forest · LOF · HBOS · Score Ensemble", "markdown"))

cells.append(cell("""\
suite  = AnomalyDetectorSuite(cfg["models"], mlflow_cfg=cfg["mlflow"])
result = suite.fit_predict(X_inj_scaled, injection.labels_true)

import pandas as pd
summary_df = pd.DataFrame(result.summary()).T.reset_index().rename(columns={"index": "Model"})
summary_df = summary_df.round(4)
print(f"\\nBest model: {result.best_model_name}  (F1={result.best.f1:.3f})")
summary_df
"""))

cells.append(cell("---\n## 5. Visualizations", "markdown"))

cells.append(cell("""\
fig = plots.plot_pca_scatter(
    X_inj_scaled, injection.labels_true,
    result.best.labels, result.best_model_name
)
plt.show()
"""))

cells.append(cell("""\
fig = plots.plot_score_distributions(
    result.models, injection.labels_true, cfg["models"]["contamination"]
)
plt.show()
"""))

cells.append(cell("""\
fig = plots.plot_model_comparison(result.summary())
plt.show()
"""))

cells.append(cell("""\
fig = plots.plot_feature_importance(
    X_inj_scaled, result.best.scores, FEATURE_NAMES, result.best_model_name
)
plt.show()
"""))

cells.append(cell("---\n## 6. HITL Precision Simulation\n12-week control-team review loop accumulating ground truth.", "markdown"))

cells.append(cell("""\
flagged_idx = np.where(result.best.labels == 1)[0]
hitl = simulate_hitl_review(
    flagged_idx,
    injection.labels_true,
    weekly_sample=eval_cfg["hitl_weekly_sample"],
    n_weeks=eval_cfg["hitl_n_weeks"],
    random_state=cfg["models"]["random_state"],
)
print(f"Final HITL precision: {hitl.final_precision:.3f}  (reviewed {hitl.total_reviewed} records)")
hitl.weekly
"""))

cells.append(cell("""\
fig = plots.plot_hitl_precision(hitl.weekly, result.best.precision, result.best_model_name)
plt.show()
"""))

cells.append(cell("---\n## 7. PSI Distribution Monitoring\nYear-over-year feature drift — flags features that need retraining attention.", "markdown"))

cells.append(cell("""\
df_09 = clean(load_raw(ROOT / cfg["data"]["path"], sheets=[cfg["data"]["sheets"][0]]))
df_10 = clean(load_raw(ROOT / cfg["data"]["path"], sheets=[cfg["data"]["sheets"][1]]))

feat_09 = build_customer_features(df_09)
feat_10 = build_customer_features(df_10)

psi_df = monitor_psi(
    feat_09, feat_10, FEATURE_NAMES,
    n_bins=eval_cfg["psi_n_bins"],
    slight_threshold=eval_cfg["psi_slight_threshold"],
    significant_threshold=eval_cfg["psi_significant_threshold"],
)
print(psi_df.to_string(index=False))
"""))

cells.append(cell("""\
fig = plots.plot_psi(psi_df)
plt.show()
"""))

cells.append(cell("---\n## 8. Final Report", "markdown"))

cells.append(cell("""\
shifted = psi_df[psi_df["status"] != "STABLE"]["feature"].tolist()
print("=" * 50)
print("FINAL REPORT")
print("=" * 50)
print(f"Customers analysed    : {len(feat_df):,}")
print(f"Synthetic anomalies   : {injection.n_injected} ({eval_cfg['synthetic_frac']:.0%})")
print(f"Best model            : {result.best_model_name}")
print(f"  Recall              : {result.best.recall:.3f}")
print(f"  Precision           : {result.best.precision:.3f}")
print(f"  F1                  : {result.best.f1:.3f}")
print(f"  ROC-AUC             : {result.best.auc:.3f}")
print(f"HITL precision (wk{eval_cfg['hitl_n_weeks']:>2}) : {hitl.final_precision:.3f}")
print(f"PSI-flagged features  : {shifted if shifted else 'None'}")
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "anomaly-detection", "language": "python", "name": "anomaly-detection"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).parent / "anomaly_detection_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {len(cells)} cells → {out}")
