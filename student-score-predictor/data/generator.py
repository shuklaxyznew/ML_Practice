"""
data/generator.py
-----------------
Synthetic student dataset generator with realistic statistical distributions,
feature correlations, and controlled noise.

Design decisions:
- Uses multivariate normal to generate correlated features (not independent sampling).
  Real students who study more also tend to attend more — this encodes that.
- Introduces a deliberate outlier group (~5%) to simulate burnout / test anxiety.
  Without this, model learns a naive "more study = higher score always" rule.
- Score is a causal function of features + irreducible Gaussian noise.
  The noise represents things no model can know: illness on exam day, etc.
- Timestamped saves for data versioning — essential for debugging drift later.
"""

import os
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class StudentDataGenerator:
    """
    Generates synthetic student performance data with realistic
    distributions and inter-feature correlations.

    Parameters
    ----------
    n_samples    : Total number of student records to generate.
    random_seed  : Seed for reproducibility. Always set this in experiments.
    outlier_frac : Fraction of students in the burnout/anxiety outlier group.
    noise_std    : Std of irreducible label noise — variance no model can explain.
    """

    FEATURE_COLS = [
        "hours_studied_per_week",
        "attendance_percentage",
        "assignments_completion_rate",
        "previous_exam_score",
        "sleep_hours_per_night",
    ]
    LABEL_COL = "final_exam_score"

    def __init__(
        self,
        n_samples: int = 2000,
        random_seed: int = 42,
        outlier_frac: float = 0.05,
        noise_std: float = 5.0,
    ):
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.outlier_frac = outlier_frac
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed=random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """
        Generate the full dataset.

        Returns a DataFrame with all feature columns, the label column,
        and a boolean 'is_outlier' flag for downstream analysis.
        """
        logger.info(
            f"Generating {self.n_samples} student records "
            f"(seed={self.random_seed}, outlier_frac={self.outlier_frac})"
        )

        n_outliers = int(self.n_samples * self.outlier_frac)
        n_normal = self.n_samples - n_outliers

        df_normal = self._generate_normal_students(n_normal)
        df_outliers = self._generate_outlier_students(n_outliers)

        df = pd.concat([df_normal, df_outliers], ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        logger.info(
            f"Dataset ready — shape: {df.shape} | "
            f"outliers: {df['is_outlier'].sum()} | "
            f"score: mean={df[self.LABEL_COL].mean():.1f}, "
            f"std={df[self.LABEL_COL].std():.1f}, "
            f"range=[{df[self.LABEL_COL].min():.1f}, {df[self.LABEL_COL].max():.1f}]"
        )
        return df

    def save(self, path: str = "data/raw/") -> str:
        """
        Generate and persist the dataset as a timestamped CSV.

        Timestamping reason: In real pipelines, data snapshots must be
        traceable. If a model degrades, you diff datasets across time
        to find distribution drift.
        """
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(path, f"student_data_{timestamp}.csv")
        df = self.generate()
        df.to_csv(filename, index=False)
        logger.info(f"Dataset saved → {filename}")
        return filename

    def correlation_report(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Print and return the feature-label correlation matrix."""
        if df is None:
            df = self.generate()
        cols = self.FEATURE_COLS + [self.LABEL_COL]
        corr = df[cols].corr().round(3)
        print("\n--- Feature Correlation Matrix ---")
        print(corr.to_string())
        return corr

    # ------------------------------------------------------------------
    # Internal: Normal student population
    # ------------------------------------------------------------------

    def _generate_normal_students(self, n: int) -> pd.DataFrame:
        """
        Generate majority population using multivariate normal distribution.

        Covariance matrix encodes domain knowledge:
          - attendance  ↔ assignments : strong positive (0.6)
          - hours       ↔ attendance  : moderate positive (0.4)
          - hours       ↔ assignments : moderate positive (0.35)
          - hours       ↔ prev_score  : mild positive (0.3)
          - sleep       ↔ everything  : weak (near-independent)
        """
        means = [15.0, 75.0, 72.0, 65.0, 7.0]

        #                          hrs    att    asgn   prev   slp
        corr = np.array([
            [1.00,  0.40,  0.35,  0.30,  0.05],   # hours_studied
            [0.40,  1.00,  0.60,  0.25,  0.05],   # attendance
            [0.35,  0.60,  1.00,  0.20,  0.05],   # assignments
            [0.30,  0.25,  0.20,  1.00,  0.10],   # previous_score
            [0.05,  0.05,  0.05,  0.10,  1.00],   # sleep
        ])

        stds = np.array([8.0, 15.0, 18.0, 15.0, 1.2])
        D = np.diag(stds)
        cov = D @ corr @ D  # convert correlation → covariance: Σ = D·R·D

        raw = self.rng.multivariate_normal(means, cov, size=n)
        df = pd.DataFrame(raw, columns=self.FEATURE_COLS)
        df = self._clip_features(df)
        df[self.LABEL_COL] = self._compute_score(df, self.noise_std)
        df[self.LABEL_COL] = df[self.LABEL_COL].clip(0, 100).round(2)
        df["is_outlier"] = False
        return df

    def _generate_outlier_students(self, n: int) -> pd.DataFrame:
        """
        Outlier group: high study hours but anomalously low scores.

        Simulates burnout, test anxiety, or high-effort-low-outcome students.
        These stress-test the model and reflect real-world distribution tails.
        Without this group, model learns a naive correlation it won't find in production.
        """
        df = pd.DataFrame()
        df["hours_studied_per_week"] = self.rng.uniform(35, 55, n)
        df["attendance_percentage"] = self.rng.uniform(60, 85, n)
        df["assignments_completion_rate"] = self.rng.uniform(55, 80, n)
        df["previous_exam_score"] = self.rng.uniform(40, 65, n)
        df["sleep_hours_per_night"] = self.rng.uniform(4, 6, n)  # sleep-deprived

        base = (
            0.10 * (df["hours_studied_per_week"] / 60 * 100)
            + 0.30 * df["attendance_percentage"]
            + 0.20 * df["assignments_completion_rate"]
            + 0.20 * df["previous_exam_score"]
        ) / 100 * 100

        noise = self.rng.normal(0, 8, n)
        df[self.LABEL_COL] = (base * 0.55 + noise).clip(20, 65).round(2)
        df["is_outlier"] = True
        return df

    def _compute_score(self, df: pd.DataFrame, noise_scale: float) -> np.ndarray:
        """
        Score = weighted combination of normalized features + Gaussian noise.

        Weights (sum to 1.0) — domain-justified:
          previous_exam_score         0.30  strongest predictor; encodes prior ability
          hours_studied_per_week      0.25  effort is second strongest signal
          attendance_percentage       0.20  classroom engagement matters
          assignments_completion_rate 0.15  consistency proxy
          sleep_hours_per_night       0.10  cognitive readiness
        """
        weights = {
            "hours_studied_per_week": 0.25,
            "attendance_percentage": 0.20,
            "assignments_completion_rate": 0.15,
            "previous_exam_score": 0.30,
            "sleep_hours_per_night": 0.10,
        }
        ranges = {
            "hours_studied_per_week": (0, 60),
            "attendance_percentage": (0, 100),
            "assignments_completion_rate": (0, 100),
            "previous_exam_score": (0, 100),
            "sleep_hours_per_night": (0, 12),
        }
        score = np.zeros(len(df))
        for col, w in weights.items():
            lo, hi = ranges[col]
            score += w * (df[col] - lo) / (hi - lo)

        score = score * 100
        noise = self.rng.normal(0, noise_scale, len(df))
        return score + noise

    def _clip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        clips = {
            "hours_studied_per_week": (0, 60),
            "attendance_percentage": (0, 100),
            "assignments_completion_rate": (0, 100),
            "previous_exam_score": (0, 100),
            "sleep_hours_per_night": (0, 12),
        }
        for col, (lo, hi) in clips.items():
            df[col] = df[col].clip(lo, hi).round(2)
        return df


if __name__ == "__main__":
    gen = StudentDataGenerator(n_samples=2000, random_seed=42)
    df = gen.generate()
    print(f"\nShape       : {df.shape}")
    print(f"Columns     : {list(df.columns)}")
    print(f"\nDescriptive Stats:\n{df.describe().round(2)}")
    gen.correlation_report(df)
    gen.save("data/raw/")
