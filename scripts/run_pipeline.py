#!/usr/bin/env python3
"""
CLI entry point — runs the full anomaly detection pipeline.

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config config/config.yaml
    python scripts/run_pipeline.py --no-mlflow
    python scripts/run_pipeline.py --save-plots outputs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from anomaly_detection import load_config
from anomaly_detection.data.loader import clean, load_raw
from anomaly_detection.evaluation.hitl import simulate_hitl_review
from anomaly_detection.evaluation.psi import monitor_psi
from anomaly_detection.evaluation.synthetic import inject_anomalies
from anomaly_detection.features.engineer import FEATURE_NAMES, build_customer_features
from anomaly_detection.models.detector import AnomalyDetectorSuite
from anomaly_detection.visualization import plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B2B Anomaly Detection Pipeline")
    p.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    p.add_argument("--save-plots", metavar="DIR", default=None,
                   help="Save all plots to this directory instead of displaying")
    return p.parse_args()


def _save_or_show(fig, path: Path | None, name: str) -> None:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / f"{name}.png", dpi=150, bbox_inches="tight")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    save_dir = Path(args.save_plots) if args.save_plots else None

    # ── 1. Load & clean ───────────────────────────────────────────────────────
    print("Loading data...")
    root = Path(args.config).parent.parent
    df_raw = load_raw(root / cfg["data"]["path"], sheets=cfg["data"]["sheets"])
    df = clean(df_raw)
    print(f"  {len(df):,} rows | {df['Customer ID'].nunique():,} customers")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("Building features...")
    feat_df = build_customer_features(df, min_slope_months=cfg["features"]["min_slope_months"])
    X_raw = feat_df[FEATURE_NAMES].values

    # ── 3. Baseline scaler (fit on raw, transform injected later) ─────────────
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    # ── 4. Synthetic injection ────────────────────────────────────────────────
    print("Injecting synthetic anomalies...")
    eval_cfg = cfg["evaluation"]
    injection = inject_anomalies(
        X_raw,
        frac=eval_cfg["synthetic_frac"],
        injection_types=eval_cfg["injection_types"],
        random_state=cfg["models"]["random_state"],
    )
    X_inj_scaled = scaler.fit_transform(imputer.fit_transform(injection.X_injected))
    print(f"  Injected {injection.n_injected} anomalies ({eval_cfg['synthetic_frac']:.0%})")

    # ── 5. Train & evaluate models ────────────────────────────────────────────
    print("Training detectors...")
    mlflow_cfg = None if args.no_mlflow else cfg["mlflow"]
    suite = AnomalyDetectorSuite(cfg["models"], mlflow_cfg=mlflow_cfg)
    result = suite.fit_predict(X_inj_scaled, injection.labels_true)

    print("\n=== Model Results ===")
    for name, row in result.summary().items():
        print(f"  {name:20s}  Recall={row['recall']:.3f}  Precision={row['precision']:.3f}"
              f"  F1={row['f1']:.3f}  AUC={row['auc']:.3f}")

    # ── 6. HITL simulation ────────────────────────────────────────────────────
    best = result.best
    flagged = np.where(best.labels == 1)[0]
    hitl = simulate_hitl_review(
        flagged,
        injection.labels_true,
        weekly_sample=eval_cfg["hitl_weekly_sample"],
        n_weeks=eval_cfg["hitl_n_weeks"],
        random_state=cfg["models"]["random_state"],
    )
    print(f"\nHITL final precision ({eval_cfg['hitl_n_weeks']} weeks): {hitl.final_precision:.3f}")

    # ── 7. PSI monitoring ─────────────────────────────────────────────────────
    df_09 = clean(load_raw(root / cfg["data"]["path"], sheets=[cfg["data"]["sheets"][0]]))
    df_10 = clean(load_raw(root / cfg["data"]["path"], sheets=[cfg["data"]["sheets"][1]]))

    feat_09 = build_customer_features(df_09)
    feat_10 = build_customer_features(df_10)
    psi_df = monitor_psi(
        feat_09, feat_10, FEATURE_NAMES,
        n_bins=eval_cfg["psi_n_bins"],
        slight_threshold=eval_cfg["psi_slight_threshold"],
        significant_threshold=eval_cfg["psi_significant_threshold"],
    )
    flagged_feats = psi_df[psi_df["status"] != "STABLE"]["feature"].tolist()
    print(f"PSI-flagged features: {flagged_feats if flagged_feats else 'None'}")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    _save_or_show(plots.plot_correlation_heatmap(feat_df.reset_index(), FEATURE_NAMES),
                  save_dir, "01_correlation")
    _save_or_show(plots.plot_pca_scatter(X_inj_scaled, injection.labels_true, best.labels, best.name),
                  save_dir, "02_pca_scatter")
    _save_or_show(plots.plot_score_distributions(result.models, injection.labels_true, cfg["models"]["contamination"]),
                  save_dir, "03_score_distributions")
    _save_or_show(plots.plot_model_comparison(result.summary()),
                  save_dir, "04_model_comparison")
    _save_or_show(plots.plot_hitl_precision(hitl.weekly, best.precision, best.name),
                  save_dir, "05_hitl_precision")
    _save_or_show(plots.plot_psi(psi_df),
                  save_dir, "06_psi")
    _save_or_show(plots.plot_feature_importance(X_inj_scaled, best.scores, FEATURE_NAMES, best.name),
                  save_dir, "07_feature_importance")

    print(f"\nDone. Best model: {result.best_model_name}  F1={best.f1:.3f}")


if __name__ == "__main__":
    main()
