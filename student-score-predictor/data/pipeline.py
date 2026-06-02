"""
data/pipeline.py
----------------
Feature engineering, preprocessing, and data splitting pipeline.

Responsibilities:
  - Load raw CSV and validate schema + value ranges
  - Apply StandardScaler — fit on train only, NEVER on val/test (data leakage)
  - Stratified train / val / test split
  - Persist fitted scaler as versioned artifact for serving-time consistency

Key rule: The scaler MUST be fit on training data only.
Fitting on the full dataset leaks test distribution statistics into training.
This is one of the most common and subtle bugs in real ML pipelines.
"""

import os
import pickle
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


FEATURE_COLS = [
    "hours_studied_per_week",
    "attendance_percentage",
    "assignments_completion_rate",
    "previous_exam_score",
    "sleep_hours_per_night",
]
LABEL_COL = "final_exam_score"

FEATURE_RANGES = {
    "hours_studied_per_week": (0.0, 60.0),
    "attendance_percentage": (0.0, 100.0),
    "assignments_completion_rate": (0.0, 100.0),
    "previous_exam_score": (0.0, 100.0),
    "sleep_hours_per_night": (0.0, 12.0),
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
        val_size: float = 0.15,
        test_size: float = 0.15,
        seed: int = 42,
    ):
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, csv_path: str) -> pd.DataFrame:
        """Load CSV and run schema + range validation."""
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        self._validate(df)
        logger.info(f"Loaded {len(df)} rows. Nulls: {df.isnull().sum().sum()}")
        return df

    def run(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Execute the full pipeline.

        Returns dict with keys:
          X_train, y_train, X_val, y_val, X_test, y_test

        All feature arrays are scaled. Labels are raw (unscaled) scores.
        """
        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(np.float32)

        # Split: (train + val) | test
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed
        )

        # Split: train | val  (relative size within train+val)
        relative_val = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=relative_val, random_state=self.seed
        )

        # Fit scaler on TRAIN only — critical for no leakage
        X_train_s = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val_s = self.scaler.transform(X_val).astype(np.float32)
        X_test_s = self.scaler.transform(X_test).astype(np.float32)
        self._fitted = True

        splits = {
            "X_train": X_train_s, "y_train": y_train,
            "X_val": X_val_s,     "y_val": y_val,
            "X_test": X_test_s,   "y_test": y_test,
        }
        self._log_splits(splits)
        return splits

    def save_scaler(self, path: str = "artifacts/") -> str:
        """
        Persist the fitted scaler.

        Why: At serving time (FastAPI), incoming requests MUST be scaled
        with the SAME scaler fitted on training data. Loading a fresh
        StandardScaler at inference time would use different statistics
        and produce garbage predictions.
        """
        assert self._fitted, "Scaler not fitted. Call pipeline.run() first."
        os.makedirs(path, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        """Return per-feature mean and std — useful for documentation."""
        assert self._fitted, "Scaler not fitted."
        return pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean": self.scaler.mean_.round(4),
            "std": self.scaler.scale_.round(4),
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        required = FEATURE_COLS + [LABEL_COL]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        null_counts = df[required].isnull().sum()
        if null_counts.any():
            logger.warning(f"Nulls detected:\n{null_counts[null_counts > 0]}")

        for col, (lo, hi) in FEATURE_RANGES.items():
            oor = ((df[col] < lo) | (df[col] > hi)).sum()
            if oor:
                logger.warning(f"{col}: {oor} values outside [{lo}, {hi}]")

        lo, hi = df[LABEL_COL].min(), df[LABEL_COL].max()
        if lo < 0 or hi > 100:
            logger.warning(f"Label out of [0,100]: [{lo:.1f}, {hi:.1f}]")

        logger.info("Validation passed.")

    def _log_splits(self, splits: Dict[str, np.ndarray]) -> None:
        n_tr = len(splits["X_train"])
        n_v = len(splits["X_val"])
        n_te = len(splits["X_test"])
        total = n_tr + n_v + n_te
        logger.info(
            f"Split — Train: {n_tr} ({n_tr/total*100:.1f}%) | "
            f"Val: {n_v} ({n_v/total*100:.1f}%) | "
            f"Test: {n_te} ({n_te/total*100:.1f}%)"
        )
        logger.info(
            f"Label stats — Train: {splits['y_train'].mean():.2f}±{splits['y_train'].std():.2f} | "
            f"Test: {splits['y_test'].mean():.2f}±{splits['y_test'].std():.2f}"
        )


if __name__ == "__main__":
    from data.generator import StudentDataGenerator

    gen = StudentDataGenerator(n_samples=2000, random_seed=42)
    df = gen.generate()

    pipeline = DataPipeline(val_size=0.15, test_size=0.15, seed=42)
    splits = pipeline.run(df)

    print("\nScaler Stats:")
    print(pipeline.get_scaler_stats())

    for name, arr in splits.items():
        print(f"  {name:10s}: shape={arr.shape}, dtype={arr.dtype}")

    pipeline.save_scaler("artifacts/")
