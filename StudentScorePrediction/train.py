"""
train.py
--------
Main training orchestrator.

Ties together: data generation → pipeline → model → training → evaluation → logging.
This is the single entry point for a full training run.

Usage:
  python train.py
  python train.py --epochs 150 --batch_size 32 --lr 5e-4
  python train.py --data_path data/raw/student_data_20240101.csv
"""

import argparse
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log"),
    ],
)
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Student Score Prediction Model")
    parser.add_argument("--data_path",    type=str,   default=None,   help="Path to existing CSV. If None, generates fresh data.")
    parser.add_argument("--n_samples",    type=int,   default=2000,   help="Number of samples to generate (if no data_path)")
    parser.add_argument("--epochs",       type=int,   default=100,    help="Max training epochs")
    parser.add_argument("--batch_size",   type=int,   default=64,     help="Batch size")
    parser.add_argument("--lr",           type=float, default=1e-3,   help="Initial learning rate")
    parser.add_argument("--dropout",      type=float, default=0.3,    help="Dropout rate")
    parser.add_argument("--patience",     type=int,   default=15,     help="Early stopping patience")
    parser.add_argument("--seed",         type=int,   default=42,     help="Random seed")
    parser.add_argument("--no_mlflow",    action="store_true",        help="Disable MLflow tracking")
    parser.add_argument("--no_feedback",  action="store_true",        help="Skip LLM feedback generation")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("STUDENT SCORE PREDICTION — TRAINING RUN")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Data
    # ------------------------------------------------------------------
    from data.generator import StudentDataGenerator
    from data.pipeline   import DataPipeline

    if args.data_path:
        logger.info(f"Loading existing data from: {args.data_path}")
        import pandas as pd
        df = pd.read_csv(args.data_path)
    else:
        logger.info(f"Generating synthetic dataset ({args.n_samples} samples)")
        gen = StudentDataGenerator(n_samples=args.n_samples, random_seed=args.seed)
        df  = gen.generate()
        gen.save("data/raw/")

    pipeline = DataPipeline(val_size=0.15, test_size=0.15, seed=args.seed)
    splits   = pipeline.run(df)
    scaler_path = pipeline.save_scaler("artifacts/")

    # ------------------------------------------------------------------
    # Step 2: Model
    # ------------------------------------------------------------------
    from model.architecture import build_model
    from model.trainer      import StudentScoreTrainer, TrainConfig

    model = build_model(
        num_features=5,
        hidden_units=(128, 64, 32),
        dropout_rate=args.dropout,
        learning_rate=args.lr,
    )
    model.summary(print_fn=logger.info)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )

    # ------------------------------------------------------------------
    # Step 3: Train
    # ------------------------------------------------------------------
    trainer = StudentScoreTrainer(model=model, config=config)
    history = trainer.train(
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
    )

    # ------------------------------------------------------------------
    # Step 4: Evaluate
    # ------------------------------------------------------------------
    test_results = trainer.evaluate(splits["X_test"], splits["y_test"])

    # ------------------------------------------------------------------
    # Step 5: MLflow Logging
    # ------------------------------------------------------------------
    if not args.no_mlflow:
        try:
            from experiments.mlflow_tracking import ExperimentTracker
            tracker = ExperimentTracker()
            with tracker:
                tracker.log_config(config)
                tracker.log_dataset_info(
                    n_train=len(splits["X_train"]),
                    n_val=len(splits["X_val"]),
                    n_test=len(splits["X_test"]),
                )
                tracker.log_history(history)
                tracker.log_test_results(test_results)
                tracker.log_scaler(scaler_path)
                tracker.log_tags({
                    "model_type":  "deep_regression",
                    "framework":   "tensorflow",
                    "data_type":   "synthetic",
                })
            logger.info(f"MLflow run: {tracker.run_id}")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    # ------------------------------------------------------------------
    # Step 6: Demo — LLM Feedback on a test student
    # ------------------------------------------------------------------
    if not args.no_feedback:
        _demo_feedback(model, pipeline, splits)

    logger.info("Training run complete.")
    return test_results


def _demo_feedback(model, pipeline, splits):
    """Generate a sample feedback report for the first test student."""
    try:
        from model.explainability    import gradient_importance
        from genai.feedback_generator import StudentFeedbackGenerator

        # Use the first test sample
        x_scaled = splits["X_test"][:1]
        score    = float(model(x_scaled, training=False).numpy().flatten()[0])
        score    = float(np.clip(score, 0, 100))

        # Inverse transform to get interpretable values
        x_raw = pipeline.scaler.inverse_transform(x_scaled)[0]

        feature_names = [
            "hours_studied_per_week",
            "attendance_percentage",
            "assignments_completion_rate",
            "previous_exam_score",
            "sleep_hours_per_night",
        ]
        student_data = dict(zip(feature_names, x_raw.tolist()))
        importance   = gradient_importance(model, x_scaled)

        gen      = StudentFeedbackGenerator()
        feedback = gen.generate(
            student_data=student_data,
            predicted_score=score,
            shap_explanation=importance,
            student_name="Demo Student",
        )

        logger.info("\n" + "="*60)
        logger.info("DEMO — LLM-GENERATED STUDENT FEEDBACK")
        logger.info("="*60)
        logger.info(f"Predicted Score: {score:.1f}/100")
        logger.info(f"\n{feedback}")

    except Exception as e:
        logger.warning(f"LLM demo skipped: {e}")


if __name__ == "__main__":
    main()
