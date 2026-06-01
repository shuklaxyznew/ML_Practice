"""
data/pipeline.py
----------------
Feature engineering, preprocessing, and data splitting pipeline.

Responsibilities:
  - Load raw CSV data
  - Validate schema and value ranges
  - Apply StandardScaler (fit on train only — never leak test stats)
  - Stratified train/val/test split
  - Persist scalers as versioned artifacts for serving-time consistency
"""

import os
import pickle
import logging
from datetime import datetime
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


FEATURE_COLS = [
    "hours_studied_per_week",
    "attendance_percentage",
    "assignments_completion_rate",
    "previous_exam_score",
    "sleep_hours_per_night",
]
LABEL_COL = "final_exam_score"

# Valid physical ranges for input validation at serving time
FEATURE_RANGES = {
    "hours_studied_per_week":      (0.0,   60.0),
    "attendance_percentage":       (0.0,  100.0),
    "assignments_completion_rate": (0.0,  100.0),
    "previous_exam_score":         (0.0,  100.0),
    "sleep_hours_per_night":       (0.0,   12.0),
}


class DataPipeline:
    """
    End-to-end preprocessing pipeline: load → validate → scale → split.

    Parameters
    ----------
    val_size  : Fraction of training data held out for validation.
    test_size : Fraction of full dataset held out for final evaluation.
    seed      : Random seed for reproducible splits.
    """

    def __init__(
        self,
        val_size:  float = 0.15,
        test_size: float = 0.15,
        seed:      int   = 42,
    ):
        self.val_size  = val_size
        self.test_size = test_size
        self.seed      = seed
        self.scaler    = StandardScaler()
        self._fitted   = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, csv_path: str) -> pd.DataFrame:
        """Load CSV and run schema + range validation."""
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        self._validate(df)
        logger.info(f"Loaded {len(df)} rows. Missing values: {df.isnull().sum().sum()}")
        return df

    def run(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Execute the full pipeline.

        Returns a dict with keys:
          X_train, y_train, X_val, y_val, X_test, y_test
        All feature arrays are scaled. Labels are unscaled (raw scores).
        """
        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(np.float32)

        # Split: train+val | test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.seed,
        )

        # Split: train | val
        relative_val = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=relative_val,
            random_state=self.seed,
        )

        # Fit scaler on TRAIN only — never on val/test (data leakage prevention)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled   = self.scaler.transform(X_val)
        X_test_scaled  = self.scaler.transform(X_test)
        self._fitted   = True

        splits = {
            "X_train": X_train_scaled.astype(np.float32),
            "y_train": y_train,
            "X_val":   X_val_scaled.astype(np.float32),
            "y_val":   y_val,
            "X_test":  X_test_scaled.astype(np.float32),
            "y_test":  y_test,
        }

        self._log_splits(splits)
        return splits

    def save_scaler(self, path: str = "artifacts/") -> str:
        """
        Persist the fitted scaler.

        Why: At serving time (FastAPI), incoming requests must be scaled
        with the SAME scaler fitted on training data. Without this, predictions
        are garbage. This is a common production bug.
        """
        assert self._fitted, "Scaler not fitted. Run pipeline.run() first."
        os.makedirs(path, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(path, f"scaler_{ts}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved → {filename}")
        return filename

    def load_scaler(self, path: str) -> None:
        """Load a previously saved scaler for serving / inference."""
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)
        self._fitted = True
        logger.info(f"Scaler loaded from: {path}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler to new inference data."""
        assert self._fitted, "Scaler not fitted."
        return self.scaler.transform(X).astype(np.float32)

    def get_scaler_stats(self) -> pd.DataFrame:
        """Return mean and std for each feature — useful for documentation."""
        assert self._fitted, "Scaler not fitted."
        return pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean":    self.scaler.mean_.round(4),
            "std":     self.scaler.scale_.round(4),
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        """Schema and range checks. Raises on hard violations, warns on soft."""
        required = FEATURE_COLS + [LABEL_COL]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        null_counts = df[required].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values detected:\n{null_counts[null_counts > 0]}")

        for col, (lo, hi) in FEATURE_RANGES.items():
            out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
            if out_of_range > 0:
                logger.warning(f"{col}: {out_of_range} values outside [{lo}, {hi}]")

        label_range = (df[LABEL_COL].min(), df[LABEL_COL].max())
        if label_range[0] < 0 or label_range[1] > 100:
            logger.warning(f"Label out of [0,100] range: {label_range}")

        logger.info("Validation complete.")

    def _log_splits(self, splits: Dict[str, np.ndarray]) -> None:
        total = (
            len(splits["X_train"]) +
            len(splits["X_val"])   +
            len(splits["X_test"])
        )
        logger.info(
            f"Split summary → "
            f"Train: {len(splits['X_train'])} ({len(splits['X_train'])/total*100:.1f}%) | "
            f"Val:   {len(splits['X_val'])}   ({len(splits['X_val'])/total*100:.1f}%) | "
            f"Test:  {len(splits['X_test'])}  ({len(splits['X_test'])/total*100:.1f}%)"
        )
        logger.info(
            f"Label stats → "
            f"Train mean: {splits['y_train'].mean():.2f} ± {splits['y_train'].std():.2f} | "
            f"Test mean:  {splits['y_test'].mean():.2f}  ± {splits['y_test'].std():.2f}"
        )


if __name__ == "__main__":
    from data.generator import StudentDataGenerator

    gen = StudentDataGenerator(n_samples=2000, random_seed=42)
    df  = gen.generate()

    pipeline = DataPipeline(val_size=0.15, test_size=0.15, seed=42)
    splits   = pipeline.run(df)

    print("\nScaler Stats:")
    print(pipeline.get_scaler_stats())

    for name, arr in splits.items():
        print(f"{name:10s}: shape={arr.shape}, dtype={arr.dtype}")

    pipeline.save_scaler(path="artifacts/")
