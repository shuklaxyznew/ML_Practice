"""
experiments/mlflow_tracking.py
-------------------------------
MLflow experiment tracking integration.

What MLflow gives you:
  - Compare runs: "did increasing dropout from 0.2 → 0.3 help?"
  - Artifact versioning: which model checkpoint goes with which dataset?
  - Reproducibility: full hyperparameter log per run
  - UI: visual loss curves, metric comparisons across experiments

This is standard in industry ML teams. Without it, you lose track of
what you've tried and why certain decisions were made.

Run the MLflow UI with:
  mlflow ui --backend-store-uri experiments/mlruns
  Then open http://localhost:5000
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import asdict

import numpy as np
import mlflow
import mlflow.tensorflow

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Wraps MLflow for clean experiment tracking.

    Parameters
    ----------
    experiment_name : MLflow experiment name (groups related runs).
    tracking_uri    : Where to store MLflow data (local or remote).
    """

    def __init__(
        self,
        experiment_name: str = "student-score-prediction",
        tracking_uri:    str = "experiments/mlruns",
    ):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = None

    # ------------------------------------------------------------------
    # Context manager: wraps a training run
    # ------------------------------------------------------------------

    def __enter__(self):
        self._run = mlflow.start_run()
        logger.info(f"MLflow run started: {self._run.info.run_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(exc_val))
        else:
            mlflow.set_tag("status", "completed")
        mlflow.end_run()
        logger.info(f"MLflow run ended: {self._run.info.run_id}")

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_config(self, config) -> None:
        """Log all training hyperparameters."""
        if hasattr(config, '__dataclass_fields__'):
            params = asdict(config)
        elif isinstance(config, dict):
            params = config
        else:
            params = vars(config)

        mlflow.log_params(params)
        logger.info(f"Logged config: {params}")

    def log_dataset_info(
        self,
        n_train: int,
        n_val:   int,
        n_test:  int,
        csv_path: Optional[str] = None,
    ) -> None:
        """Log dataset statistics for reproducibility."""
        mlflow.log_params({
            "data.n_train": n_train,
            "data.n_val":   n_val,
            "data.n_test":  n_test,
            "data.total":   n_train + n_val + n_test,
        })
        if csv_path:
            mlflow.set_tag("data.source", csv_path)

    def log_history(self, history: Dict[str, List[float]]) -> None:
        """
        Log per-epoch metrics from the training loop.
        MLflow stores these as time-series — visible as charts in the UI.
        """
        for metric_name, values in history.items():
            for epoch, value in enumerate(values, start=1):
                mlflow.log_metric(metric_name, value, step=epoch)

        # Also log best values as summary params
        if "val_loss" in history:
            mlflow.log_metric("best_val_loss", min(history["val_loss"]))
            mlflow.log_metric("best_epoch",
                              int(np.argmin(history["val_loss"])) + 1)

    def log_test_results(self, results: Dict[str, float]) -> None:
        """Log final test set evaluation metrics."""
        for k, v in results.items():
            mlflow.log_metric(f"test.{k}", v)
        logger.info(f"Test results logged: {results}")

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log the trained TF model as an MLflow artifact."""
        mlflow.tensorflow.log_model(model, artifact_path=artifact_path)
        logger.info(f"Model logged to MLflow artifact: {artifact_path}")

    def log_scaler(self, scaler_path: str) -> None:
        """Log the scaler pickle as an artifact."""
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

    def log_feature_importance(self, importance_df) -> None:
        """Log feature importance as a JSON artifact."""
        importance_dict = importance_df.set_index("feature")["mean_abs_shap"].to_dict()
        tmp_path = "/tmp/feature_importance.json"
        with open(tmp_path, "w") as f:
            json.dump(importance_dict, f, indent=2)
        mlflow.log_artifact(tmp_path, artifact_path="analysis")

    def log_tags(self, tags: Dict[str, str]) -> None:
        """Log arbitrary tags — useful for marking model type, architecture, etc."""
        mlflow.set_tags(tags)

    @property
    def run_id(self) -> Optional[str]:
        return self._run.info.run_id if self._run else None


# ------------------------------------------------------------------
# Standalone run logger (for one-off logging outside context manager)
# ------------------------------------------------------------------

def log_run(
    config,
    history: Dict[str, List[float]],
    test_results: Dict[str, float],
    model=None,
    tags: Optional[Dict[str, str]] = None,
    experiment_name: str = "student-score-prediction",
) -> str:
    """
    Convenience function: log a complete run in one call.
    Returns the run_id.
    """
    tracker = ExperimentTracker(experiment_name=experiment_name)
    with tracker:
        tracker.log_config(config)
        tracker.log_history(history)
        tracker.log_test_results(test_results)
        if model is not None:
            tracker.log_model(model)
        if tags:
            tracker.log_tags(tags)
    return tracker.run_id


if __name__ == "__main__":
    # Smoke test
    from model.trainer import TrainConfig
    config = TrainConfig(epochs=50, batch_size=64, learning_rate=1e-3)

    dummy_history = {
        "train_loss": [10.0, 8.5, 7.2, 6.1, 5.3],
        "val_loss":   [11.0, 9.0, 7.8, 6.9, 6.5],
        "train_mae":  [2.5, 2.2, 2.0, 1.8, 1.6],
        "val_mae":    [2.7, 2.4, 2.2, 2.1, 2.0],
        "lr":         [1e-3, 9e-4, 8e-4, 7e-4, 6e-4],
    }
    dummy_test = {"mse": 35.2, "mae": 4.1, "rmse": 5.93, "r2": 0.87}

    run_id = log_run(
        config=config,
        history=dummy_history,
        test_results=dummy_test,
        tags={"model_type": "deep_regression", "dataset": "synthetic_2000"},
    )
    print(f"\nRun logged. ID: {run_id}")
    print("View at: mlflow ui --backend-store-uri experiments/mlruns")
