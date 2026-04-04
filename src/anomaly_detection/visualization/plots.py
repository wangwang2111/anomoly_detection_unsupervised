"""
visualization.plots
-------------------
All plot functions take data arguments, auto-save to assets/, and return
the matplotlib Figure so callers can still call plt.show() in a notebook.

Set SAVE_DIR before calling any plot function to override the default:
    import anomaly_detection.visualization.plots as plots
    plots.SAVE_DIR = Path("my_output_dir")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid", palette="muted")

# ── Auto-save directory (relative to repo root) ───────────────────────────────
SAVE_DIR: Path = Path(__file__).parents[4] / "assets"

Fig = matplotlib.figure.Figure


def _save(fig: Fig, name: str) -> None:
    """Save figure to SAVE_DIR/<name>.png at 150 dpi."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_correlation_heatmap(feature_df: pd.DataFrame, features: list[str]) -> Fig:
    """Correlation matrix of raw (unscaled) features."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = feature_df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_correlation")
    return fig


def plot_pca_scatter(
    X_scaled: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    model_name: str,
    random_state: int = 42,
) -> Fig:
    """Side-by-side PCA scatter: ground truth vs model predictions."""
    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (title, col) in zip(axes, [
        ("Ground Truth (Injected)", labels_true),
        (f"{model_name} Predictions", labels_pred),
    ]):
        ax.scatter(X2[col == 0, 0], X2[col == 0, 1], c="steelblue", alpha=0.4, s=15, label="Normal")
        ax.scatter(X2[col == 1, 0], X2[col == 1, 1], c="tomato", alpha=0.9, s=40, marker="x", label="Anomaly")
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.legend(markerscale=1.5)

    fig.suptitle("PCA Projection — Anomaly Detection Results", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_pca_scatter")
    return fig


def plot_score_distributions(
    model_results: dict,
    labels_true: np.ndarray,
    contamination: float,
) -> Fig:
    """Anomaly score histograms for each individual model."""
    model_names = [n for n in model_results if n != "Ensemble"]
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        scores = model_results[name].scores
        thresh = np.percentile(scores, 100 * (1 - contamination))
        ax.hist(scores[labels_true == 0], bins=50, alpha=0.6, color="steelblue", label="Normal", density=True)
        ax.hist(scores[labels_true == 1], bins=30, alpha=0.7, color="tomato",    label="Injected", density=True)
        ax.axvline(thresh, color="black", linestyle="--", linewidth=1.5, label="Threshold")
        ax.set_title(name)
        ax.set_xlabel("Anomaly Score")
        ax.legend(fontsize=8)

    fig.suptitle("Anomaly Score Distribution — Injected vs Normal", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_score_distributions")
    return fig


def plot_model_comparison(summary: dict[str, dict[str, float]]) -> Fig:
    """Grouped bar chart comparing Recall / Precision / F1 / PR-AUC across models."""
    df = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "Model"})
    metric_cols = ["recall", "precision", "f1", "auc"]
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    x = np.arange(len(df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (col, c) in enumerate(zip(metric_cols, colors)):
        label = "PR-AUC" if col == "auc" else col.upper().replace("_", "-")
        ax.bar(x + i * width, df[col], width, label=label, color=c, alpha=0.85)

    # Annotate F1 bars
    for i, row in df.iterrows():
        ax.text(x[i] + 2 * width, row["f1"] + 0.015, f"{row['f1']:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#59a14f")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Synthetic Anomaly Injection", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, "04_model_comparison")
    return fig


def plot_hitl_precision(hitl_df: pd.DataFrame, static_precision: float, model_name: str) -> Fig:
    """Cumulative HITL precision curve over review weeks."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hitl_df["week"], hitl_df["cumulative_precision"],
            marker="o", color="steelblue", linewidth=2)
    ax.axhline(static_precision, color="tomato", linestyle="--",
               label=f"Model static precision ({static_precision:.2f})")
    ax.fill_between(hitl_df["week"], hitl_df["cumulative_precision"], alpha=0.12, color="steelblue")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative Precision")
    ax.set_title(f"HITL Precision Accumulation — {model_name} flags reviewed by control team")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, "05_hitl_precision")
    return fig


def plot_psi(psi_df: pd.DataFrame) -> Fig:
    """Horizontal bar chart of PSI values with threshold lines."""
    colors = [
        "tomato" if p > 0.2 else ("gold" if p > 0.1 else "mediumseagreen")
        for p in psi_df["psi"]
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(psi_df["feature"], psi_df["psi"], color=colors)
    ax.axvline(0.1, color="gold",   linestyle="--", linewidth=1.5, label="Slight shift (0.1)")
    ax.axvline(0.2, color="tomato", linestyle="--", linewidth=1.5, label="Significant shift (0.2)")
    ax.set_xlabel("PSI")
    ax.set_title("Population Stability Index — Baseline vs Monitoring", fontsize=12, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, "06_psi")
    return fig


def plot_feature_importance(X_scaled: np.ndarray, scores: np.ndarray, feature_names: list[str], model_name: str) -> Fig:
    """Correlation between each feature and the anomaly score (proxy for importance)."""
    importances = pd.Series(
        [abs(float(np.corrcoef(X_scaled[:, i], scores)[0, 1])) for i in range(len(feature_names))],
        index=feature_names,
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="steelblue", alpha=0.85)
    ax.set_xlabel("|Correlation with Anomaly Score|")
    ax.set_title(f"Feature Importance ({model_name})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "07_feature_importance")
    return fig
