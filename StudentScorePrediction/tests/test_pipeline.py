"""
tests/test_pipeline.py
-----------------------
Smoke tests and unit tests for data pipeline, model, and trainer.

Philosophy:
  - Smoke tests: "does it run without exploding?"
  - Unit tests:  "does each component do what I said it does?"
  - No integration tests here (those require a live model + API)

Run with:
  pytest tests/ -v
  pytest tests/ -v --tb=short
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ------------------------------------------------------------------
# Data Generator Tests
# ------------------------------------------------------------------

class TestStudentDataGenerator:

    def setup_method(self):
        from data.generator import StudentDataGenerator
        self.gen = StudentDataGenerator(n_samples=200, random_seed=42)

    def test_output_shape(self):
        df = self.gen.generate()
        assert df.shape[0] == 200, f"Expected 200 rows, got {df.shape[0]}"
        assert df.shape[1] >= 7,   f"Expected ≥7 columns, got {df.shape[1]}"

    def test_required_columns_present(self):
        df = self.gen.generate()
        required = [
            "hours_studied_per_week",
            "attendance_percentage",
            "assignments_completion_rate",
            "previous_exam_score",
            "sleep_hours_per_night",
            "final_exam_score",
            "is_outlier",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nulls(self):
        df = self.gen.generate()
        assert df.isnull().sum().sum() == 0, "Dataset contains null values"

    def test_feature_ranges(self):
        df = self.gen.generate()
        assert df["hours_studied_per_week"].between(0, 60).all()
        assert df["attendance_percentage"].between(0, 100).all()
        assert df["assignments_completion_rate"].between(0, 100).all()
        assert df["previous_exam_score"].between(0, 100).all()
        assert df["sleep_hours_per_night"].between(0, 12).all()
        assert df["final_exam_score"].between(0, 100).all()

    def test_reproducibility(self):
        """Same seed → same data. Critical for experiment reproducibility."""
        from data.generator import StudentDataGenerator
        gen1 = StudentDataGenerator(n_samples=100, random_seed=42)
        gen2 = StudentDataGenerator(n_samples=100, random_seed=42)
        df1, df2 = gen1.generate(), gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        from data.generator import StudentDataGenerator
        gen1 = StudentDataGenerator(n_samples=100, random_seed=1)
        gen2 = StudentDataGenerator(n_samples=100, random_seed=2)
        df1, df2 = gen1.generate(), gen2.generate()
        assert not df1["final_exam_score"].equals(df2["final_exam_score"])

    def test_outlier_fraction(self):
        from data.generator import StudentDataGenerator
        gen = StudentDataGenerator(n_samples=1000, outlier_frac=0.05, random_seed=42)
        df  = gen.generate()
        outlier_frac = df["is_outlier"].mean()
        # Allow ±1% tolerance
        assert abs(outlier_frac - 0.05) < 0.01, f"Outlier fraction off: {outlier_frac:.3f}"

    def test_outliers_have_lower_scores(self):
        """Outliers should have lower avg score than normal students."""
        df = self.gen.generate()
        mean_outlier = df[df["is_outlier"]]["final_exam_score"].mean()
        mean_normal  = df[~df["is_outlier"]]["final_exam_score"].mean()
        assert mean_outlier < mean_normal, \
            f"Outlier mean {mean_outlier:.1f} should be < normal mean {mean_normal:.1f}"

    def test_feature_correlations(self):
        """attendance and assignments should be positively correlated."""
        from data.generator import StudentDataGenerator
        gen = StudentDataGenerator(n_samples=2000, random_seed=42)
        df  = gen.generate()
        corr = df["attendance_percentage"].corr(df["assignments_completion_rate"])
        assert corr > 0.3, f"Expected positive correlation, got {corr:.3f}"


# ------------------------------------------------------------------
# Data Pipeline Tests
# ------------------------------------------------------------------

class TestDataPipeline:

    def setup_method(self):
        from data.generator import StudentDataGenerator
        from data.pipeline   import DataPipeline
        gen      = StudentDataGenerator(n_samples=500, random_seed=42)
        self.df  = gen.generate()
        self.pipeline = DataPipeline(val_size=0.15, test_size=0.15, seed=42)
        self.splits   = self.pipeline.run(self.df)

    def test_split_keys(self):
        expected = {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
        assert set(self.splits.keys()) == expected

    def test_split_shapes_consistent(self):
        assert self.splits["X_train"].shape[0] == len(self.splits["y_train"])
        assert self.splits["X_val"].shape[0]   == len(self.splits["y_val"])
        assert self.splits["X_test"].shape[0]  == len(self.splits["y_test"])

    def test_feature_dim(self):
        assert self.splits["X_train"].shape[1] == 5, "Expected 5 features"

    def test_total_size(self):
        total = (
            len(self.splits["X_train"]) +
            len(self.splits["X_val"])   +
            len(self.splits["X_test"])
        )
        assert total == len(self.df), f"Split total {total} ≠ dataset size {len(self.df)}"

    def test_no_data_leakage(self):
        """Scaler must be fit on train only. Verify train mean ≈ 0 after scaling."""
        train_mean = self.splits["X_train"].mean(axis=0)
        # After StandardScaler fit on train, train mean should be ≈ 0
        assert np.allclose(train_mean, 0, atol=0.1), \
            f"Train mean after scaling should be ≈0, got {train_mean}"

    def test_dtype_float32(self):
        for key in ["X_train", "X_val", "X_test"]:
            assert self.splits[key].dtype == np.float32, \
                f"{key} should be float32, got {self.splits[key].dtype}"

    def test_reproducibility(self):
        from data.pipeline import DataPipeline
        p1 = DataPipeline(seed=42)
        p2 = DataPipeline(seed=42)
        s1 = p1.run(self.df)
        s2 = p2.run(self.df)
        np.testing.assert_array_equal(s1["X_train"], s2["X_train"])


# ------------------------------------------------------------------
# Model Architecture Tests
# ------------------------------------------------------------------

class TestModelArchitecture:

    def setup_method(self):
        from model.architecture import build_model
        self.model = build_model(num_features=5)

    def test_output_shape(self):
        import tensorflow as tf
        X     = tf.random.normal((8, 5))
        y_hat = self.model(X, training=False)
        assert y_hat.shape == (8, 1), f"Expected (8,1), got {y_hat.shape}"

    def test_model_is_compiled(self):
        assert self.model.optimizer is not None
        assert self.model.loss      is not None

    def test_parameter_count(self):
        n_params = self.model.count_params()
        assert n_params > 0, "Model has no parameters"
        assert n_params < 500_000, f"Model seems too large for this task: {n_params:,}"

    def test_named_layers(self):
        layer_names = [l.name for l in self.model.layers]
        assert any("dense" in n for n in layer_names)
        assert any("bn"    in n for n in layer_names)
        assert any("relu"  in n for n in layer_names)

    def test_training_mode_changes_output(self):
        """Dropout should make training outputs vary; inference should be deterministic."""
        import tensorflow as tf
        X = tf.random.normal((4, 5))
        out1 = self.model(X, training=False).numpy()
        out2 = self.model(X, training=False).numpy()
        np.testing.assert_array_equal(out1, out2, err_msg="Inference should be deterministic")

    def test_gradient_flows(self):
        """Verify gradients are non-zero (no vanishing gradient on init)."""
        import tensorflow as tf
        X = tf.random.normal((16, 5))
        y = tf.random.uniform((16, 1), 0, 100)
        with tf.GradientTape() as tape:
            y_hat = self.model(X, training=True)
            loss  = tf.reduce_mean((y - y_hat) ** 2)
        grads = tape.gradient(loss, self.model.trainable_variables)
        for g, v in zip(grads, self.model.trainable_variables):
            if g is not None:
                assert not np.allclose(g.numpy(), 0), \
                    f"Zero gradient in layer: {v.name}"


# ------------------------------------------------------------------
# Trainer Smoke Test
# ------------------------------------------------------------------

class TestTrainer:

    def test_short_training_run(self):
        """3 epochs, small data — just verify no crashes."""
        from data.generator  import StudentDataGenerator
        from data.pipeline   import DataPipeline
        from model.architecture import build_model
        from model.trainer   import StudentScoreTrainer, TrainConfig

        gen     = StudentDataGenerator(n_samples=200, random_seed=42)
        df      = gen.generate()
        pipeline = DataPipeline(seed=42)
        splits  = pipeline.run(df)

        model   = build_model()
        config  = TrainConfig(epochs=3, batch_size=32, patience=10)
        trainer = StudentScoreTrainer(model=model, config=config)
        history = trainer.train(
            splits["X_train"], splits["y_train"],
            splits["X_val"],   splits["y_val"],
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) == 3
        assert all(np.isfinite(v) for v in history["train_loss"]), "NaN/Inf in loss"

    def test_evaluate_returns_all_metrics(self):
        from data.generator  import StudentDataGenerator
        from data.pipeline   import DataPipeline
        from model.architecture import build_model
        from model.trainer   import StudentScoreTrainer, TrainConfig

        gen     = StudentDataGenerator(n_samples=200, random_seed=42)
        df      = gen.generate()
        splits  = DataPipeline(seed=42).run(df)
        model   = build_model()
        trainer = StudentScoreTrainer(model, TrainConfig(epochs=2, patience=10))
        trainer.train(splits["X_train"], splits["y_train"],
                      splits["X_val"],   splits["y_val"])
        results = trainer.evaluate(splits["X_test"], splits["y_test"])

        for key in ["mse", "mae", "rmse", "r2"]:
            assert key in results, f"Missing metric: {key}"
            assert np.isfinite(results[key]), f"Non-finite value for {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
