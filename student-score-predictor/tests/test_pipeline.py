"""
tests/test_pipeline.py
-----------------------
Unit and smoke tests for data, model, and trainer layers.

Run:
  pytest tests/ -v
  pytest tests/ -v --tb=short
  pytest tests/ -v --cov=. --cov-report=html

Test philosophy:
  - Smoke tests : does it run without exploding?
  - Unit tests  : does each component do what I claimed it does?
  - No LLM tests: those require a live backend (integration tests, run separately)
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ======================================================================
# Data Generator
# ======================================================================

class TestStudentDataGenerator:

    def setup_method(self):
        from data.generator import StudentDataGenerator
        self.gen = StudentDataGenerator(n_samples=300, random_seed=42)

    def test_output_shape(self):
        df = self.gen.generate()
        assert df.shape[0] == 300

    def test_required_columns(self):
        df = self.gen.generate()
        expected = [
            "hours_studied_per_week",
            "attendance_percentage",
            "assignments_completion_rate",
            "previous_exam_score",
            "sleep_hours_per_night",
            "final_exam_score",
            "is_outlier",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nulls(self):
        df = self.gen.generate()
        assert df.isnull().sum().sum() == 0, "Dataset must contain zero nulls"

    def test_feature_ranges(self):
        df = self.gen.generate()
        assert df["hours_studied_per_week"].between(0, 60).all(),      "hours_studied out of range"
        assert df["attendance_percentage"].between(0, 100).all(),      "attendance out of range"
        assert df["assignments_completion_rate"].between(0, 100).all(),"assignments out of range"
        assert df["previous_exam_score"].between(0, 100).all(),        "previous_score out of range"
        assert df["sleep_hours_per_night"].between(0, 12).all(),       "sleep out of range"
        assert df["final_exam_score"].between(0, 100).all(),           "label out of range"

    def test_reproducibility(self):
        """Critical: same seed must produce identical datasets."""
        from data.generator import StudentDataGenerator
        g1 = StudentDataGenerator(n_samples=100, random_seed=99)
        g2 = StudentDataGenerator(n_samples=100, random_seed=99)
        pd.testing.assert_frame_equal(g1.generate(), g2.generate())

    def test_different_seeds_differ(self):
        from data.generator import StudentDataGenerator
        g1 = StudentDataGenerator(n_samples=200, random_seed=1)
        g2 = StudentDataGenerator(n_samples=200, random_seed=2)
        assert not g1.generate()["final_exam_score"].equals(g2.generate()["final_exam_score"])

    def test_outlier_fraction(self):
        from data.generator import StudentDataGenerator
        gen = StudentDataGenerator(n_samples=1000, outlier_frac=0.05, random_seed=42)
        df  = gen.generate()
        frac = df["is_outlier"].mean()
        assert abs(frac - 0.05) < 0.015, f"Outlier fraction off target: {frac:.3f}"

    def test_outliers_score_lower_than_normal(self):
        """Outlier group must have lower average score — validates the generation logic."""
        df = self.gen.generate()
        mean_out = df[df["is_outlier"]]["final_exam_score"].mean()
        mean_nor = df[~df["is_outlier"]]["final_exam_score"].mean()
        assert mean_out < mean_nor, (
            f"Outlier mean ({mean_out:.1f}) should be below normal mean ({mean_nor:.1f})"
        )

    def test_attendance_assignments_correlated(self):
        """Attendance and assignments must be positively correlated (encoded in covariance matrix)."""
        from data.generator import StudentDataGenerator
        gen = StudentDataGenerator(n_samples=2000, random_seed=42)
        df  = gen.generate()
        corr = df["attendance_percentage"].corr(df["assignments_completion_rate"])
        assert corr > 0.3, f"Expected corr > 0.3, got {corr:.3f}"

    def test_label_is_function_of_features(self):
        """Score must be positively correlated with study hours."""
        from data.generator import StudentDataGenerator
        gen = StudentDataGenerator(n_samples=2000, random_seed=42)
        df  = gen.generate()
        corr = df["hours_studied_per_week"].corr(df["final_exam_score"])
        assert corr > 0.2, f"Expected positive correlation with score, got {corr:.3f}"


# ======================================================================
# Data Pipeline
# ======================================================================

class TestDataPipeline:

    def setup_method(self):
        from data.generator import StudentDataGenerator
        from data.pipeline   import DataPipeline

        gen        = StudentDataGenerator(n_samples=500, random_seed=42)
        self.df    = gen.generate()
        self.pipe  = DataPipeline(val_size=0.15, test_size=0.15, seed=42)
        self.splits = self.pipe.run(self.df)

    def test_split_keys_present(self):
        expected = {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
        assert set(self.splits.keys()) == expected

    def test_no_size_mismatch(self):
        assert len(self.splits["X_train"]) == len(self.splits["y_train"])
        assert len(self.splits["X_val"])   == len(self.splits["y_val"])
        assert len(self.splits["X_test"])  == len(self.splits["y_test"])

    def test_five_features(self):
        for k in ["X_train", "X_val", "X_test"]:
            assert self.splits[k].shape[1] == 5, f"{k} must have 5 features"

    def test_sizes_sum_to_total(self):
        total = (
            len(self.splits["X_train"]) +
            len(self.splits["X_val"])   +
            len(self.splits["X_test"])
        )
        assert total == len(self.df), "Split sizes must sum to full dataset"

    def test_no_data_leakage(self):
        """
        After StandardScaler.fit(X_train), training set mean must be ≈ 0.
        If scaler were fit on full data, this assertion would be weaker.
        """
        train_mean = self.splits["X_train"].mean(axis=0)
        assert np.allclose(train_mean, 0, atol=0.15), (
            f"Train mean should be ≈ 0 after scaling. Got: {train_mean}"
        )

    def test_float32_dtype(self):
        for k in ["X_train", "X_val", "X_test"]:
            assert self.splits[k].dtype == np.float32, f"{k} must be float32"

    def test_reproducible_splits(self):
        from data.pipeline import DataPipeline
        p1 = DataPipeline(seed=42)
        p2 = DataPipeline(seed=42)
        s1 = p1.run(self.df)
        s2 = p2.run(self.df)
        np.testing.assert_array_equal(s1["X_train"], s2["X_train"])

    def test_scaler_stats(self):
        stats = self.pipe.get_scaler_stats()
        assert len(stats) == 5
        assert "feature" in stats.columns
        assert "mean" in stats.columns
        assert "std" in stats.columns


# ======================================================================
# Model Architecture
# ======================================================================

class TestModelArchitecture:

    def setup_method(self):
        from model.architecture import build_model
        self.model = build_model(num_features=5, hidden_units=(64, 32), dropout_rate=0.2)

    def test_output_shape(self):
        import tensorflow as tf
        X = tf.random.normal((8, 5))
        out = self.model(X, training=False)
        assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"

    def test_compiled(self):
        assert self.model.optimizer is not None
        assert self.model.loss is not None

    def test_param_count_reasonable(self):
        n = self.model.count_params()
        assert n > 100,     "Model seems too small"
        assert n < 500_000, f"Model too large for this task: {n:,} params"

    def test_layer_naming(self):
        names = [l.name for l in self.model.layers]
        assert any("dense"   in n for n in names), "Missing dense layer"
        assert any("bn"      in n for n in names), "Missing batch norm"
        assert any("dropout" in n for n in names), "Missing dropout"

    def test_inference_deterministic(self):
        """With training=False, same input must always produce same output."""
        import tensorflow as tf
        X    = tf.random.normal((4, 5))
        out1 = self.model(X, training=False).numpy()
        out2 = self.model(X, training=False).numpy()
        np.testing.assert_array_equal(out1, out2)

    def test_output_unbounded(self):
        """Regression output must not be clipped by sigmoid/softmax."""
        import tensorflow as tf
        # Feed extreme inputs — sigmoid would cap at ~0 or ~1
        X   = tf.constant([[10.0] * 5, [-10.0] * 5])
        out = self.model(X, training=False).numpy().flatten()
        # Values should differ substantially if layer is linear
        assert abs(out[0] - out[1]) > 0.01, "Output activation may not be linear"

    def test_gradient_flow(self):
        """Verify no vanishing gradients on initialisation (all-zero gradients = dead network)."""
        import tensorflow as tf
        X = tf.random.normal((16, 5))
        y = tf.random.uniform((16, 1), 0, 100)
        with tf.GradientTape() as tape:
            y_hat = self.model(X, training=True)
            loss  = tf.reduce_mean((y - y_hat) ** 2)
        grads = tape.gradient(loss, self.model.trainable_variables)
        for g, v in zip(grads, self.model.trainable_variables):
            if g is not None:
                assert not np.allclose(g.numpy(), 0), (
                    f"Zero gradient in {v.name} — possible vanishing gradient"
                )

    def test_wide_variant_builds(self):
        from model.architecture import build_model_wide
        model = build_model_wide()
        assert model.count_params() > 0


# ======================================================================
# Trainer (smoke tests — lightweight, no full training)
# ======================================================================

class TestTrainer:

    def _make_splits(self, n: int = 300):
        from data.generator import StudentDataGenerator
        from data.pipeline   import DataPipeline

        gen    = StudentDataGenerator(n_samples=n, random_seed=42)
        df     = gen.generate()
        return DataPipeline(seed=42).run(df)

    def test_three_epoch_run_no_crash(self):
        """3 epochs, small data — just verify no exceptions thrown."""
        from model.architecture import build_model
        from model.trainer      import StudentScoreTrainer, TrainConfig

        splits  = self._make_splits(200)
        model   = build_model(hidden_units=(32, 16))
        trainer = StudentScoreTrainer(model, TrainConfig(epochs=3, batch_size=32, patience=99))
        history = trainer.train(
            splits["X_train"], splits["y_train"],
            splits["X_val"],   splits["y_val"],
        )
        assert len(history["train_loss"]) == 3

    def test_loss_is_finite(self):
        from model.architecture import build_model
        from model.trainer      import StudentScoreTrainer, TrainConfig

        splits  = self._make_splits(200)
        model   = build_model(hidden_units=(32, 16))
        trainer = StudentScoreTrainer(model, TrainConfig(epochs=3, batch_size=32, patience=99))
        history = trainer.train(
            splits["X_train"], splits["y_train"],
            splits["X_val"],   splits["y_val"],
        )
        assert all(np.isfinite(v) for v in history["train_loss"]), "NaN/Inf in training loss"
        assert all(np.isfinite(v) for v in history["val_loss"]),   "NaN/Inf in val loss"

    def test_evaluate_returns_all_metrics(self):
        from model.architecture import build_model
        from model.trainer      import StudentScoreTrainer, TrainConfig

        splits  = self._make_splits(200)
        model   = build_model(hidden_units=(32, 16))
        trainer = StudentScoreTrainer(model, TrainConfig(epochs=2, batch_size=32, patience=99))
        trainer.train(
            splits["X_train"], splits["y_train"],
            splits["X_val"],   splits["y_val"],
        )
        results = trainer.evaluate(splits["X_test"], splits["y_test"])

        for key in ["mse", "mae", "rmse", "r2"]:
            assert key in results,               f"Missing metric: {key}"
            assert np.isfinite(results[key]),    f"Non-finite value for: {key}"

    def test_early_stopping_triggers(self):
        """With patience=1, stopping must happen before max epochs."""
        from model.architecture import build_model
        from model.trainer      import StudentScoreTrainer, TrainConfig

        splits  = self._make_splits(200)
        model   = build_model(hidden_units=(32, 16))
        trainer = StudentScoreTrainer(model, TrainConfig(epochs=50, batch_size=32, patience=1))
        history = trainer.train(
            splits["X_train"], splits["y_train"],
            splits["X_val"],   splits["y_val"],
        )
        assert len(history["train_loss"]) < 50, "Early stopping should have triggered before 50 epochs"


# ======================================================================
# Explainability
# ======================================================================

class TestExplainability:

    def setup_method(self):
        from model.architecture import build_model
        self.model = build_model(hidden_units=(32, 16))
        self.X = np.random.randn(20, 5).astype(np.float32)

    def test_gradient_importance_keys(self):
        from model.explainability import gradient_importance
        result = gradient_importance(self.model, self.X)
        assert len(result) == 5, "Must have one entry per feature"

    def test_gradient_importance_values_finite(self):
        from model.explainability import gradient_importance
        result = gradient_importance(self.model, self.X)
        for feat, val in result.items():
            assert np.isfinite(val), f"Non-finite importance for {feat}"

    def test_gradient_importance_normalized(self):
        """Values should sum to ~1.0 (normalized in the implementation)."""
        from model.explainability import gradient_importance
        result = gradient_importance(self.model, self.X)
        total = sum(result.values())
        assert abs(total - 1.0) < 0.05, f"Importances should sum to ~1, got {total:.3f}"


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
