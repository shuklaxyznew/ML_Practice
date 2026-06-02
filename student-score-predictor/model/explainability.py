"""
model/explainability.py
-----------------------
Feature attribution: SHAP (primary) and gradient-based (fallback).

Why explainability in production?
  - Regulators (GDPR, EU AI Act) require explainable predictions
  - Teachers/counselors need to know WHY a student is predicted low
  - Debugging: are features contributing as domain knowledge expects?
  - This output feeds directly into the GenAI feedback layer

SHAP value interpretation:
  Positive SHAP → feature pushed prediction UP vs the average student
  Negative SHAP → feature pushed prediction DOWN vs the average student
  Magnitude     → how much it moved the prediction

Example:
  Average score (base): 72
  Predicted score:      54

  SHAP breakdown:
    sleep_hours:        -9.8  (only 4hrs — severe deficit)
    previous_score:     -6.4  (scored 45 last time)
    attendance:         -4.1  (missed 30% of classes)
    hours_studied:      +2.3  (studied well)
    assignments:        +0.0  (neutral)
    ─────────────────────────
    Sum of SHAPs:       -18   → 72 - 18 ≈ 54  ✓
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Using gradient fallback. Install: pip install shap")


FEATURE_COLS = [
    "hours_studied_per_week",
    "attendance_percentage",
    "assignments_completion_rate",
    "previous_exam_score",
    "sleep_hours_per_night",
]

FEATURE_LABELS = {
    "hours_studied_per_week":      "Weekly Study Hours",
    "attendance_percentage":       "Class Attendance",
    "assignments_completion_rate": "Assignment Completion",
    "previous_exam_score":         "Previous Exam Score",
    "sleep_hours_per_night":       "Sleep Hours",
}


class ModelExplainer:
    """
    SHAP-based explainer for the trained Keras model.

    Uses DeepExplainer, which leverages a background dataset to compute
    expected values and approximates Shapley values via DeepLIFT.

    Parameters
    ----------
    model        : Trained Keras model.
    X_background : Sample of training data used as reference distribution.
                   100-200 representative samples is sufficient.
    """

    def __init__(self, model: tf.keras.Model, X_background: np.ndarray):
        self.model = model
        self.X_background = X_background[:200]
        self.explainer = None
        self._initialized = False

    def initialize(self) -> None:
        """Build the SHAP explainer. Slow first call (~30s), fast afterward."""
        if not SHAP_AVAILABLE:
            raise ImportError("Install SHAP: pip install shap")
        logger.info("Initializing SHAP DeepExplainer...")
        self.explainer = shap.DeepExplainer(self.model, self.X_background)
        self._initialized = True
        logger.info("SHAP explainer ready.")

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a batch of inputs.

        Returns array of shape (n_samples, n_features).
        Each value = feature's contribution to prediction vs the base value.
        """
        assert self._initialized, "Call initialize() first."
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return np.array(shap_values).squeeze()

    def explain_single(self, x: np.ndarray) -> Dict[str, float]:
        """
        Explain one student prediction.

        Returns {feature_label: shap_value}, sorted by absolute impact desc.
        """
        x = np.atleast_2d(x).astype(np.float32)
        shap_vals = self.explain(x)
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]

        result = {
            FEATURE_LABELS.get(col, col): float(shap_vals[i])
            for i, col in enumerate(FEATURE_COLS)
        }
        return dict(sorted(result.items(), key=lambda kv: abs(kv[1]), reverse=True))

    def global_importance(self, X: np.ndarray) -> pd.DataFrame:
        """
        Global feature importance = mean |SHAP| across all samples.

        This is the SHAP equivalent of tree-model feature importance,
        but it works correctly for neural networks.
        """
        shap_vals = self.explain(X)
        df = pd.DataFrame({
            "feature": [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        logger.info("\n--- Global Feature Importance (mean |SHAP|) ---")
        for _, row in df.iterrows():
            bar = "█" * int(row["mean_abs_shap"] * 2)
            logger.info(f"  {row['feature']:30s}: {row['mean_abs_shap']:.4f}  {bar}")

        return df

    def format_for_llm(
        self,
        prediction: float,
        shap_dict: Dict[str, float],
    ) -> str:
        """
        Format SHAP results into structured text for the GenAI feedback layer.
        This is the bridge between ML output and LLM input.
        """
        up   = [(k, v) for k, v in shap_dict.items() if v > 0]
        down = [(k, v) for k, v in shap_dict.items() if v < 0]

        lines = [f"Predicted Score: {prediction:.1f}/100", ""]
        lines.append("Positive factors (boosting score):")
        for feat, val in sorted(up, key=lambda x: x[1], reverse=True):
            lines.append(f"  + {feat}: +{val:.2f} pts")

        lines.append("\nNegative factors (reducing score):")
        for feat, val in sorted(down, key=lambda x: x[1]):
            lines.append(f"  - {feat}: {val:.2f} pts")

        return "\n".join(lines)


# ======================================================================
# Gradient-based fallback (no SHAP dependency)
# ======================================================================

def gradient_importance(
    model: tf.keras.Model,
    X: np.ndarray,
) -> Dict[str, float]:
    """
    Gradient-based feature importance: ∂output/∂input averaged over samples.

    Less theoretically grounded than SHAP but:
      - Zero extra dependencies
      - Fast (single forward + backward pass)
      - Good enough for the LLM prompt context

    Used as fallback when SHAP is not installed or for real-time API calls.
    """
    X_tensor = tf.Variable(X.astype(np.float32))
    with tf.GradientTape() as tape:
        predictions = model(X_tensor, training=False)
    grads = tape.gradient(predictions, X_tensor).numpy()

    importance = np.abs(grads).mean(axis=0)
    result = {
        FEATURE_LABELS.get(col, col): float(importance[i])
        for i, col in enumerate(FEATURE_COLS)
    }
    # Normalize to sum to 1 for interpretability
    total = sum(result.values()) or 1.0
    result = {k: round(v / total, 4) for k, v in result.items()}
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


def get_feature_attribution(
    model: tf.keras.Model,
    X: np.ndarray,
    X_background: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Unified entry point: uses SHAP if available, gradient otherwise.

    Parameters
    ----------
    model        : Trained Keras model.
    X            : Input sample(s) to explain. Shape (n, n_features).
    X_background : Background data for SHAP. If None, falls back to gradients.
    """
    if SHAP_AVAILABLE and X_background is not None:
        try:
            explainer = ModelExplainer(model, X_background)
            explainer.initialize()
            x = np.atleast_2d(X[0])
            return explainer.explain_single(x)
        except Exception as e:
            logger.warning(f"SHAP failed ({e}), falling back to gradients.")

    return gradient_importance(model, np.atleast_2d(X))


if __name__ == "__main__":
    from model.architecture import build_model

    model = build_model()
    X_dummy = np.random.randn(50, 5).astype(np.float32)

    print("\nGradient-based importance (untrained model — values are random):")
    importance = gradient_importance(model, X_dummy)
    for feat, val in importance.items():
        bar = "█" * int(val * 40)
        print(f"  {feat:30s}: {val:.4f}  {bar}")
