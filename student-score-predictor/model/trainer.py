"""
model/trainer.py
----------------
Custom training loop using tf.GradientTape.

Why not model.fit()?
  - Full gradient access: inspect, clip, or modify gradients per step
  - Fine-grained control: essential for LoRA, multi-task losses, custom schedulers
  - Transparency: you see exactly what happens at each forward/backward pass
  - This is how HuggingFace Trainer and LoRA fine-tuning scripts work under the hood

The GradientTape pattern (memorise this — it's the foundation of LLM fine-tuning):
  1. Forward pass inside tape context    → tape records all operations
  2. Compute loss from predictions       → scalar value
  3. tape.gradient(loss, variables)      → compute ∂loss/∂weights (backprop)
  4. optimizer.apply_gradients(...)      → update weights: w = w - lr × ∂loss/∂w

Connection to LoRA:
  In LoRA you freeze base weights and only train injected rank matrices.
  Implementation: filter model.trainable_variables to only the LoRA params,
  then pass that subset to tape.gradient(). Same loop, different variable list.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """
    All training hyperparameters in one place.
    Dataclass → serializable, easy to log to MLflow, easy to reproduce runs.
    """
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    min_lr: float = 1e-5
    patience: int = 15          # early stopping patience (epochs without improvement)
    grad_clip_norm: float = 1.0  # gradient clipping threshold
    checkpoint_dir: str = "checkpoints/"
    log_every_n: int = 5        # print metrics every N epochs
    lr_schedule: str = "cosine"  # "cosine" | "constant"


class StudentScoreTrainer:
    """
    Custom GradientTape training loop for the regression model.

    Handles:
      - Per-step gradient computation and clipping
      - Per-epoch metric tracking (stateful Keras metrics)
      - Validation loop (no gradients, training=False)
      - Early stopping with best-weight restoration
      - Model checkpointing on val_loss improvement
      - Cosine LR decay schedule
    """

    def __init__(self, model: keras.Model, config: TrainConfig):
        self.model = model
        self.config = config
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_mae": [], "train_rmse": [],
            "val_loss": [],   "val_mae": [],   "val_rmse": [],
            "lr": [],
        }
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_weights = None
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns the history dict for plotting and MLflow logging.
        """
        train_ds = self._make_dataset(X_train, y_train, shuffle=True)
        val_ds = self._make_dataset(X_val, y_val, shuffle=False)

        optimizer = self._build_optimizer()
        loss_fn = tf.keras.losses.MeanSquaredError()

        # Stateful metrics — reset at start of each epoch
        metrics = {
            "train_loss": tf.keras.metrics.Mean(name="train_loss"),
            "train_mae":  tf.keras.metrics.MeanAbsoluteError(name="train_mae"),
            "train_rmse": tf.keras.metrics.RootMeanSquaredError(name="train_rmse"),
            "val_loss":   tf.keras.metrics.Mean(name="val_loss"),
            "val_mae":    tf.keras.metrics.MeanAbsoluteError(name="val_mae"),
            "val_rmse":   tf.keras.metrics.RootMeanSquaredError(name="val_rmse"),
        }

        logger.info(
            f"Training started — epochs={self.config.epochs}, "
            f"batch_size={self.config.batch_size}, "
            f"lr={self.config.learning_rate}, "
            f"schedule={self.config.lr_schedule}"
        )
        t0 = time.time()

        for epoch in range(1, self.config.epochs + 1):

            # ── Training pass ─────────────────────────────────────────
            for X_batch, y_batch in train_ds:
                self._train_step(X_batch, y_batch, optimizer, loss_fn, metrics)

            # ── Validation pass ────────────────────────────────────────
            for X_batch, y_batch in val_ds:
                self._val_step(X_batch, y_batch, loss_fn, metrics)

            # ── Record history ─────────────────────────────────────────
            current_lr = float(optimizer.learning_rate)
            self._record(metrics, current_lr)

            # ── Logging ────────────────────────────────────────────────
            if epoch % self.config.log_every_n == 0 or epoch == 1:
                self._log_epoch(epoch, metrics, current_lr)

            # ── Checkpoint on improvement ──────────────────────────────
            val_loss = float(metrics["val_loss"].result())
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_weights = self.model.get_weights()
                self._patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self._patience_counter += 1

            # ── Reset all metrics for next epoch ──────────────────────
            for m in metrics.values():
                m.reset_state()

            # ── Early stopping ─────────────────────────────────────────
            if self._patience_counter >= self.config.patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_loss={self._best_val_loss:.4f}"
                )
                break

        elapsed = time.time() - t0
        logger.info(
            f"Training complete in {elapsed:.1f}s. "
            f"Best val_loss={self._best_val_loss:.4f}"
        )

        # Restore best weights before returning
        if self._best_weights is not None:
            self.model.set_weights(self._best_weights)
            logger.info("Best weights restored.")

        return self.history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Final evaluation on the held-out test set.

        Call this ONCE at the very end — never during hyperparameter tuning.
        Any evaluation during tuning must use the validation set only.
        """
        y_pred = self.model(X_test, training=False).numpy().flatten()

        mse  = float(np.mean((y_pred - y_test) ** 2))
        mae  = float(np.mean(np.abs(y_pred - y_test)))
        rmse = float(np.sqrt(mse))

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot)

        results = {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

        logger.info("=" * 55)
        logger.info("TEST SET EVALUATION (final, unseen data)")
        logger.info(f"  MSE  : {mse:.4f}  (avg squared error in score² units)")
        logger.info(f"  RMSE : {rmse:.4f} (avg error in score points — interpretable)")
        logger.info(f"  MAE  : {mae:.4f}  (avg absolute error in score points)")
        logger.info(f"  R²   : {r2:.4f}  (variance explained; 1.0 = perfect)")
        logger.info("=" * 55)

        return results

    # ------------------------------------------------------------------
    # Internal: Single train step — the GradientTape core
    # ------------------------------------------------------------------

    @tf.function   # compile to TF graph for performance
    def _train_step(self, X_batch, y_batch, optimizer, loss_fn, metrics):
        """
        Core GradientTape pattern.

        The tape records every operation inside its context block.
        tape.gradient() then traverses the recorded graph backward
        to compute ∂loss/∂trainable_variables (automatic differentiation).

        This is identical in structure to LoRA fine-tuning —
        the only difference is which variables you pass to tape.gradient().
        """
        with tf.GradientTape() as tape:
            y_pred = self.model(X_batch, training=True)   # forward pass
            loss = loss_fn(y_batch, y_pred)               # MSE loss
            loss += sum(self.model.losses)                # L2 regularization terms

        # Compute ∂loss/∂w for every trainable weight
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Gradient clipping: rescale if global L2 norm exceeds threshold.
        # Prevents a single bad batch from destroying weights.
        # Standard in Transformer training (clip_norm=1.0 is the default in most papers).
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip_norm)

        # Weight update: w = w - lr × gradient
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update stateful metrics
        metrics["train_loss"].update_state(loss)
        metrics["train_mae"].update_state(y_batch, y_pred)
        metrics["train_rmse"].update_state(y_batch, y_pred)

    @tf.function
    def _val_step(self, X_batch, y_batch, loss_fn, metrics):
        """Validation step — no tape, no weight updates, training=False."""
        y_pred = self.model(X_batch, training=False)
        loss = loss_fn(y_batch, y_pred)
        metrics["val_loss"].update_state(loss)
        metrics["val_mae"].update_state(y_batch, y_pred)
        metrics["val_rmse"].update_state(y_batch, y_pred)

    # ------------------------------------------------------------------
    # Internal: Utilities
    # ------------------------------------------------------------------

    def _make_dataset(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
    ) -> tf.data.Dataset:
        """
        tf.data pipeline: shuffle → batch → prefetch.
        AUTOTUNE lets TensorFlow optimize buffer sizes at runtime.
        """
        ds = tf.data.Dataset.from_tensor_slices((
            X.astype(np.float32),
            y.astype(np.float32),
        ))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X), seed=42)
        return ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

    def _build_optimizer(self):
        if self.config.lr_schedule == "cosine":
            schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=self.config.epochs * 30,
                alpha=self.config.min_lr / self.config.learning_rate,
            )
            return tf.keras.optimizers.Adam(learning_rate=schedule)
        return tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

    def _record(self, metrics: dict, lr: float) -> None:
        self.history["train_loss"].append(float(metrics["train_loss"].result()))
        self.history["train_mae"].append(float(metrics["train_mae"].result()))
        self.history["train_rmse"].append(float(metrics["train_rmse"].result()))
        self.history["val_loss"].append(float(metrics["val_loss"].result()))
        self.history["val_mae"].append(float(metrics["val_mae"].result()))
        self.history["val_rmse"].append(float(metrics["val_rmse"].result()))
        self.history["lr"].append(lr)

    def _log_epoch(self, epoch: int, metrics: dict, lr: float) -> None:
        logger.info(
            f"Epoch {epoch:4d} | "
            f"train_loss={metrics['train_loss'].result():.4f}  "
            f"train_mae={metrics['train_mae'].result():.4f} | "
            f"val_loss={metrics['val_loss'].result():.4f}  "
            f"val_mae={metrics['val_mae'].result():.4f} | "
            f"lr={lr:.2e}"
        )

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = os.path.join(
            self.config.checkpoint_dir,
            f"best_epoch{epoch:03d}_valloss{val_loss:.4f}",
        )
        self.model.save(path)
        logger.info(f"Checkpoint → {path}")
