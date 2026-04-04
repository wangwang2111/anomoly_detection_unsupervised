"""
models.detector
---------------
Uniform train/predict/score API wrapping four PyOD detectors and an
average-score ensemble.  All model hyperparameters flow in via the
config dict — no magic numbers in this file.

Models: IsolationForest, LOF, COPOD, ECOD.
HBOS was removed — COPOD and ECOD handle feature correlations correctly
whereas HBOS assumes independence, which degrades accuracy on this dataset.

The :class:`AnomalyDetectorSuite` trains all models in one call,
returns a :class:`DetectionResult` dataclass, and logs every run to
MLflow automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow
import mlflow.sklearn
import numpy as np
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


@dataclass
class ModelResult:
    name: str
    labels: np.ndarray        # 0 = normal, 1 = anomaly
    scores: np.ndarray        # higher = more anomalous
    recall: float
    precision: float
    f1: float
    auc: float
    model: object = field(repr=False)


@dataclass
class DetectionResult:
    models: dict[str, ModelResult]
    ensemble_labels: np.ndarray
    ensemble_scores: np.ndarray
    ensemble_recall: float
    ensemble_precision: float
    ensemble_f1: float
    ensemble_auc: float
    best_model_name: str

    @property
    def best(self) -> ModelResult:
        return self.models[self.best_model_name]

    def summary(self) -> dict[str, dict[str, float]]:
        rows = {
            name: {
                "recall": r.recall,
                "precision": r.precision,
                "f1": r.f1,
                "auc": r.auc,
            }
            for name, r in self.models.items()
        }
        rows["Ensemble"] = {
            "recall": self.ensemble_recall,
            "precision": self.ensemble_precision,
            "f1": self.ensemble_f1,
            "auc": self.ensemble_auc,
        }
        return rows


class AnomalyDetectorSuite:
    """Train all three detectors + ensemble on a scaled feature matrix.

    Parameters
    ----------
    cfg:
        ``config["models"]`` sub-dict from config.yaml.
    mlflow_cfg:
        ``config["mlflow"]`` sub-dict.  Pass ``None`` to skip MLflow logging.
    """

    def __init__(self, cfg: dict, mlflow_cfg: dict | None = None) -> None:
        self.cfg = cfg
        self.mlflow_cfg = mlflow_cfg
        self._contamination = cfg["contamination"]
        self._rng = cfg["random_state"]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_models(self) -> dict[str, object]:
        return {
            "IsolationForest": IForest(
                contamination=self._contamination,
                random_state=self._rng,
                n_estimators=self.cfg["isolation_forest"]["n_estimators"],
            ),
            "LOF": LOF(
                contamination=self._contamination,
                n_neighbors=self.cfg["lof"]["n_neighbors"],
            ),
            "COPOD": COPOD(
                contamination=self._contamination,
            ),
            "ECOD": ECOD(
                contamination=self._contamination,
            ),
        }

    def _score_model(
        self,
        name: str,
        model,
        X: np.ndarray,
        labels_true: np.ndarray,
    ) -> ModelResult:
        model.fit(X)
        preds = model.labels_
        scores = model.decision_scores_
        return ModelResult(
            name=name,
            labels=preds,
            scores=scores,
            recall=recall_score(labels_true, preds, zero_division=0),
            precision=precision_score(labels_true, preds, zero_division=0),
            f1=f1_score(labels_true, preds, zero_division=0),
            auc=average_precision_score(labels_true, scores),
            model=model,
        )

    def _log_to_mlflow(self, result: ModelResult, n_customers: int, n_features: int) -> None:
        if self.mlflow_cfg is None:
            return
        mlflow.set_tracking_uri(self.mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(self.mlflow_cfg["experiment_name"])
        with mlflow.start_run(run_name=result.name):
            mlflow.log_params({
                "model": result.name,
                "contamination": self._contamination,
                "n_customers": n_customers,
                "n_features": n_features,
            })
            mlflow.log_metrics({
                "recall": result.recall,
                "precision": result.precision,
                "f1": result.f1,
                "pr_auc": result.auc,
            })
            mlflow.sklearn.log_model(result.model, name=result.name)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_predict(
        self,
        X: np.ndarray,
        labels_true: np.ndarray,
    ) -> DetectionResult:
        """Fit all detectors, compute metrics, log to MLflow, return results.

        Parameters
        ----------
        X:
            Scaled feature matrix (n_customers × n_features).
        labels_true:
            Ground-truth labels (0 = normal, 1 = anomaly) for metric computation.
            In production without labels, pass ``np.zeros(len(X), dtype=int)``
            — metrics will be meaningless but the detector still runs.

        Returns
        -------
        :class:`DetectionResult`
        """
        model_defs = self._build_models()
        results: dict[str, ModelResult] = {}

        for name, model in model_defs.items():
            r = self._score_model(name, model, X, labels_true)
            self._log_to_mlflow(r, n_customers=len(X), n_features=X.shape[1])
            results[name] = r

        # ── Ensemble: average normalised decision scores ──────────────────────
        score_matrix = np.column_stack([results[m].scores for m in results])
        # Min-max normalise per model before averaging
        lo = score_matrix.min(axis=0)
        hi = score_matrix.max(axis=0)
        norm = (score_matrix - lo) / (hi - lo + 1e-9)
        ens_scores = norm.mean(axis=1)

        threshold = np.percentile(ens_scores, 100 * (1 - self._contamination))
        ens_labels = (ens_scores >= threshold).astype(int)

        best_name = max(results, key=lambda m: results[m].f1)

        ens_recall    = recall_score(labels_true, ens_labels, zero_division=0)
        ens_precision = precision_score(labels_true, ens_labels, zero_division=0)
        ens_f1        = f1_score(labels_true, ens_labels, zero_division=0)
        ens_auc       = average_precision_score(labels_true, ens_scores)

        if self.mlflow_cfg is not None:
            mlflow.set_tracking_uri(self.mlflow_cfg["tracking_uri"])
            mlflow.set_experiment(self.mlflow_cfg["experiment_name"])
            with mlflow.start_run(run_name="Ensemble"):
                mlflow.log_params({
                    "model": "Ensemble",
                    "contamination": self._contamination,
                    "n_customers": len(X),
                    "n_features": X.shape[1],
                })
                mlflow.log_metrics({
                    "recall": ens_recall,
                    "precision": ens_precision,
                    "f1": ens_f1,
                    "pr_auc": ens_auc,
                })

        return DetectionResult(
            models=results,
            ensemble_labels=ens_labels,
            ensemble_scores=ens_scores,
            ensemble_recall=ens_recall,
            ensemble_precision=ens_precision,
            ensemble_f1=ens_f1,
            ensemble_auc=ens_auc,
            best_model_name=best_name,
        )
