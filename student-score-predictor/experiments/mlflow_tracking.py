"""
experiments/mlflow_tracking.py
-------------------------------
MLflow experiment tracking for all training runs.

Why MLflow?
  Without tracking you will:
    - Run 50 experiments and forget which dropout=0.2 run produced R²=0.91
    - Lose the exact scaler used with a specific model checkpoint
    - Be unable to reproduce a result 2 weeks later

  With MLflow you get:
    - Full hyperparameter log per run (epochs, lr, batch_size, dropout...)
    - Per-epoch loss curves (train_loss, val_loss, MAE, RMSE)
    - Saved model artifact tied to the exact run
    - Scaler artifact (same version as model — essential for serving)
    - Visual comparison across runs in the MLflow UI

Run the UI:
  mlflow ui --backend-store-uri experiments/mlruns
  Open http://localhost:5000
"""

import json
import logging
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import mlflow
import mlflow.tensorflow
import numpy as np

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Wraps MLflow for clean, consistent experiment logging.

    Usage (context manager — recommended):
      with ExperimentTracker() as tracker:
          tracker.log_config(config)
          tracker.log_history(history)
          tracker.log_test_results(results)

    Parameters
    ----------
    experiment_name : MLflow experiment name (groups related runs).
    tracking_uri    : Where MLflow stores data (local path or remote URI).
    """

    def __init__(
        self,
        experiment_name: str = "student-score-prediction",
        tracking_uri: str = "experiments/mlruns",
    ):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self._run = mlflow.start_run()
        logger.info(f"MLflow run started: {self._run.info.run_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type else "completed"
        mlflow.set_tag("status", status)
        if exc_type:
            mlflow.set_tag("error", str(exc_val))
        mlflow.end_run()
        logger.info(f"MLflow run ended ({status}): {self._run.info.run_id}")

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_config(self, config) -> None:
        """Log all training hyperparameters."""
        if hasattr(config, "__dataclass_fields__"):
            params = asdict(config)
        elif isinstance(config, dict):
            params = config
        else:
            params = vars(config)
        mlflow.log_params(params)
        logger.info(f"Logged hyperparameters: {params}")

    def log_dataset_info(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        csv_path: Optional[str] = None,
    ) -> None:
        """Log dataset sizes for reproducibility auditing."""
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
        Log per-epoch training metrics as time series.

        MLflow stores these as step-indexed values — visible as line charts in the UI.
        Use the UI to see overfitting (val_loss diverging from train_loss).
        """
        for metric_name, values in history.items():
            for epoch, value in enumerate(values, start=1):
                mlflow.log_metric(metric_name, float(value), step=epoch)

        # Summary metrics for easy sorting/filtering in the UI
        if "val_loss" in history and history["val_loss"]:
            mlflow.log_metric("best_val_loss", min(history["val_loss"]))
            mlflow.log_metric(
                "best_epoch",
                int(np.argmin(history["val_loss"])) + 1,
            )

    def log_test_results(self, results: Dict[str, float]) -> None:
        """Log final held-out test set evaluation metrics."""
        for k, v in results.items():
            mlflow.log_metric(f"test.{k}", float(v))
        logger.info(f"Test results logged: {results}")

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log the trained TensorFlow model as a versioned MLflow artifact."""
        mlflow.tensorflow.log_model(model, artifact_path=artifact_path)
        logger.info(f"Model artifact logged: {artifact_path}")

    def log_scaler(self, scaler_path: str) -> None:
        """Log the scaler pickle — must be tied to the same run as the model."""
        if os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
            logger.info(f"Scaler artifact logged: {scaler_path}")

    def log_feature_importance(self, importance: dict) -> None:
        """Log feature importance as a JSON artifact for downstream analysis."""
        tmp = "/tmp/feature_importance.json"
        with open(tmp, "w") as f:
            json.dump(importance, f, indent=2)
        mlflow.log_artifact(tmp, artifact_path="analysis")

    def log_tags(self, tags: Dict[str, str]) -> None:
        mlflow.set_tags(tags)

    @property
    def run_id(self) -> Optional[str]:
        return self._run.info.run_id if self._run else None


# ======================================================================
# Convenience function for one-call logging
# ======================================================================

def log_run(
    config,
    history: Dict[str, List[float]],
    test_results: Dict[str, float],
    model=None,
    scaler_path: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    experiment_name: str = "student-score-prediction",
) -> str:
    """Log a complete training run in one call. Returns run_id."""
    tracker = ExperimentTracker(experiment_name=experiment_name)
    with tracker:
        tracker.log_config(config)
        tracker.log_history(history)
        tracker.log_test_results(test_results)
        if model is not None:
            tracker.log_model(model)
        if scaler_path:
            tracker.log_scaler(scaler_path)
        if tags:
            tracker.log_tags(tags)
    return tracker.run_id


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    from model.trainer import TrainConfig

    config = TrainConfig(epochs=50, batch_size=64, learning_rate=1e-3)
    dummy_history = {
        "train_loss": [10.0, 8.5, 7.2, 6.1, 5.3],
        "val_loss":   [11.0, 9.0, 7.8, 6.9, 6.5],
        "train_mae":  [2.5,  2.2, 2.0, 1.8, 1.6],
        "val_mae":    [2.7,  2.4, 2.2, 2.1, 2.0],
        "lr":         [1e-3, 9e-4, 8e-4, 7e-4, 6e-4],
    }
    dummy_test = {"mse": 35.2, "mae": 4.1, "rmse": 5.93, "r2": 0.87}

    run_id = log_run(
        config=config,
        history=dummy_history,
        test_results=dummy_test,
        tags={"model_type": "deep_regression", "dataset": "synthetic_2000"},
    )
    print(f"\nRun ID: {run_id}")
    print("View: mlflow ui --backend-store-uri experiments/mlruns")
