"""
model/explainability.py
-----------------------
SHAP (SHapley Additive exPlanations) integration for model interpretability.

Why SHAP in production?
  - Regulators (GDPR, EU AI Act) require explainable predictions
  - Teachers/counselors need to know WHY a student is predicted to score low
  - Debugging: are features contributing as expected?
  - This feeds directly into the GenAI feedback layer

SHAP value interpretation:
  - Positive SHAP → this feature pushed the prediction UP
  - Negative SHAP → this feature pushed the prediction DOWN
  - Magnitude → how much it pushed

Example:
  Student predicted score: 58
  Base value: 72
  SHAP values:
    sleep_hours:        -8.2  (only 5hrs sleep — major negative)
    hours_studied:      +3.1  (studied well)
    previous_score:     -6.4  (poor history)
    attendance:         -2.8  (missed classes)
    assignments:        +0.3  (submitted most assignments)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")


FEATURE_COLS = [
    "hours_studied_per_week",
    "attendance_percentage",
    "assignments_completion_rate",
    "previous_exam_score",
    "sleep_hours_per_night",
]

# Human-readable labels for reports
FEATURE_LABELS = {
    "hours_studied_per_week":      "Weekly Study Hours",
    "attendance_percentage":       "Class Attendance",
    "assignments_completion_rate": "Assignment Completion",
    "previous_exam_score":         "Previous Exam Score",
    "sleep_hours_per_night":       "Sleep Hours",
}


class ModelExplainer:
    """
    Wraps a trained Keras model with SHAP explanation capabilities.

    Uses DeepExplainer for neural networks — it uses a background
    dataset to compute expected values and approximate Shapley values
    via DeepLIFT.

    Parameters
    ----------
    model        : Trained Keras model.
    X_background : Training data sample used as reference distribution.
                   Typically 100–500 representative samples.
    """

    def __init__(self, model: tf.keras.Model, X_background: np.ndarray):
        self.model        = model
        self.X_background = X_background[:200]  # 200 background samples is sufficient
        self.explainer    = None
        self._initialized = False

    def initialize(self) -> None:
        """Build the SHAP explainer. Slow first call, fast subsequent explains."""
        if not SHAP_AVAILABLE:
            raise ImportError("Install SHAP: pip install shap")

        logger.info("Initializing SHAP DeepExplainer (first-time setup, ~30s)...")
        self.explainer    = shap.DeepExplainer(self.model, self.X_background)
        self._initialized = True
        logger.info("SHAP explainer ready.")

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a batch of inputs.

        Returns array of shape (n_samples, n_features).
        Each value is the feature's contribution to the prediction
        relative to the expected (base) value.
        """
        assert self._initialized, "Call initialize() first."
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # regression: single output
        return shap_values.squeeze()

    def explain_single(self, x: np.ndarray) -> Dict[str, float]:
        """
        Explain a single student prediction.

        Returns a dict mapping feature name → SHAP value,
        sorted by absolute importance descending.

        Parameters
        ----------
        x : 1D or 2D array of shape (n_features,) or (1, n_features)
        """
        x = np.atleast_2d(x).astype(np.float32)
        shap_vals = self.explain(x)[0] if len(x) == 1 else self.explain(x)

        result = {
            FEATURE_LABELS.get(col, col): float(shap_vals[i])
            for i, col in enumerate(FEATURE_COLS)
        }
        # Sort by absolute impact
        result = dict(sorted(result.items(), key=lambda kv: abs(kv[1]), reverse=True))
        return result

    def global_importance(self, X: np.ndarray) -> pd.DataFrame:
        """
        Compute global feature importance as mean |SHAP| across samples.

        This is the SHAP equivalent of feature importance from tree models —
        but it works correctly for neural networks.
        """
        shap_vals = self.explain(X)
        importance = pd.DataFrame({
            "feature":    [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        logger.info("\n--- Global Feature Importance (mean |SHAP|) ---")
        for _, row in importance.iterrows():
            bar = "█" * int(row["mean_abs_shap"] * 2)
            logger.info(f"  {row['feature']:30s}: {row['mean_abs_shap']:.4f}  {bar}")

        return importance

    def format_for_llm(
        self,
        x_raw: np.ndarray,
        prediction: float,
        shap_dict: Dict[str, float],
    ) -> str:
        """
        Format SHAP explanation into a structured context string
        for the GenAI feedback generator.

        This bridges the ML layer with the LLM layer.
        """
        drivers_up   = [(k, v) for k, v in shap_dict.items() if v > 0]
        drivers_down = [(k, v) for k, v in shap_dict.items() if v < 0]

        lines = [
            f"Predicted Score: {prediction:.1f}/100",
            "",
            "Positive Factors (boosting predicted score):",
        ]
        for feat, val in sorted(drivers_up, key=lambda x: x[1], reverse=True):
            lines.append(f"  + {feat}: +{val:.2f} points")

        lines.append("\nNegative Factors (reducing predicted score):")
        for feat, val in sorted(drivers_down, key=lambda x: x[1]):
            lines.append(f"  - {feat}: {val:.2f} points")

        return "\n".join(lines)


# ------------------------------------------------------------------
# Fallback: Gradient-based importance (no SHAP dependency)
# ------------------------------------------------------------------

def gradient_importance(
    model: tf.keras.Model,
    X: np.ndarray,
) -> Dict[str, float]:
    """
    Gradient-based feature importance as fallback when SHAP is unavailable.

    Computes ∂output/∂input averaged over samples.
    Less precise than SHAP but requires no extra dependencies.
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
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


if __name__ == "__main__":
    # Quick smoke test with a dummy model
    from model.architecture import build_model
    model = build_model()

    X_dummy = np.random.randn(200, 5).astype(np.float32)
    grad_imp = gradient_importance(model, X_dummy)
    print("\nGradient-based importance (untrained model — random):")
    for feat, val in grad_imp.items():
        print(f"  {feat:30s}: {val:.4f}")
