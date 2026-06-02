"""
train.py
--------
Main entry point: orchestrates the full training pipeline.

  data generation → pipeline → model → training → evaluation → MLflow → demo

Usage:
  python train.py                                  # defaults
  python train.py --epochs 150 --lr 5e-4          # custom hyperparams
  python train.py --data_path data/raw/XYZ.csv    # use existing dataset
  python train.py --no_mlflow --no_feedback       # minimal run
"""

import argparse
import logging
import os
import sys

import numpy as np

os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log"),
    ],
)
logger = logging.getLogger(__name__)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Student Score Prediction Model")
    p.add_argument("--data_path",   type=str,   default=None,   help="Path to existing CSV. If omitted, generates fresh data.")
    p.add_argument("--n_samples",   type=int,   default=2000,   help="Samples to generate (if no --data_path)")
    p.add_argument("--epochs",      type=int,   default=100,    help="Max training epochs")
    p.add_argument("--batch_size",  type=int,   default=64,     help="Batch size")
    p.add_argument("--lr",          type=float, default=1e-3,   help="Initial learning rate")
    p.add_argument("--dropout",     type=float, default=0.3,    help="Dropout rate")
    p.add_argument("--patience",    type=int,   default=15,     help="Early stopping patience")
    p.add_argument("--seed",        type=int,   default=42,     help="Random seed")
    p.add_argument("--no_mlflow",   action="store_true",        help="Disable MLflow tracking")
    p.add_argument("--no_feedback", action="store_true",        help="Skip LLM feedback demo")
    return p.parse_args()


# ======================================================================
# Main pipeline
# ======================================================================

def main() -> dict:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("STUDENT SCORE PREDICTION — TRAINING RUN")
    logger.info("=" * 60)
    logger.info(
        f"Config: epochs={args.epochs}, batch={args.batch_size}, "
        f"lr={args.lr}, dropout={args.dropout}, seed={args.seed}"
    )

    # ── Step 1: Data ──────────────────────────────────────────────────
    from data.generator import StudentDataGenerator
    from data.pipeline   import DataPipeline

    if args.data_path:
        import pandas as pd
        logger.info(f"Loading existing dataset: {args.data_path}")
        df = pd.read_csv(args.data_path)
    else:
        logger.info(f"Generating synthetic dataset ({args.n_samples} samples)...")
        gen = StudentDataGenerator(n_samples=args.n_samples, random_seed=args.seed)
        df  = gen.generate()
        gen.save("data/raw/")

    pipeline = DataPipeline(val_size=0.15, test_size=0.15, seed=args.seed)
    splits   = pipeline.run(df)
    scaler_path = pipeline.save_scaler("artifacts/")

    logger.info(
        f"Data ready — train={len(splits['X_train'])}, "
        f"val={len(splits['X_val'])}, test={len(splits['X_test'])}"
    )

    # ── Step 2: Model ─────────────────────────────────────────────────
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

    # ── Step 3: Train ─────────────────────────────────────────────────
    trainer = StudentScoreTrainer(model=model, config=config)
    history = trainer.train(
        X_train=splits["X_train"], y_train=splits["y_train"],
        X_val=splits["X_val"],     y_val=splits["y_val"],
    )

    # ── Step 4: Evaluate on test set ──────────────────────────────────
    # This is called ONCE. Never use test set for hyperparameter decisions.
    test_results = trainer.evaluate(splits["X_test"], splits["y_test"])

    # ── Step 5: Feature attribution ───────────────────────────────────
    from model.explainability import gradient_importance
    attribution = gradient_importance(model, splits["X_train"][:100])
    logger.info(f"Feature attribution: {attribution}")

    # ── Step 6: MLflow ────────────────────────────────────────────────
    if not args.no_mlflow:
        _log_to_mlflow(config, history, test_results, model, scaler_path, attribution)

    # ── Step 7: LLM feedback demo ─────────────────────────────────────
    if not args.no_feedback:
        _feedback_demo(model, pipeline, splits)

    logger.info("Training run complete.")
    return test_results


# ======================================================================
# Helpers
# ======================================================================

def _log_to_mlflow(config, history, test_results, model, scaler_path, attribution):
    try:
        from experiments.mlflow_tracking import ExperimentTracker

        with ExperimentTracker() as tracker:
            tracker.log_config(config)
            tracker.log_history(history)
            tracker.log_test_results(test_results)
            tracker.log_scaler(scaler_path)
            tracker.log_feature_importance(attribution)
            tracker.log_tags({
                "model_type": "deep_regression",
                "framework":  "tensorflow",
                "data_type":  "synthetic",
            })
        logger.info("MLflow logging complete.")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")


def _feedback_demo(model, pipeline, splits):
    """Generate and print a sample LLM feedback report for the first test student."""
    try:
        from model.explainability    import gradient_importance
        from genai.feedback_generator import StudentFeedbackGenerator

        x_scaled = splits["X_test"][:1]
        score    = float(np.clip(
            model(x_scaled, training=False).numpy().flatten()[0], 0, 100
        ))

        # Inverse-transform to get interpretable raw values
        x_raw = pipeline.scaler.inverse_transform(x_scaled)[0]
        feature_names = [
            "hours_studied_per_week",
            "attendance_percentage",
            "assignments_completion_rate",
            "previous_exam_score",
            "sleep_hours_per_night",
        ]
        student_data = dict(zip(feature_names, x_raw.tolist()))
        attribution  = gradient_importance(model, x_scaled)

        gen      = StudentFeedbackGenerator()
        feedback = gen.generate(
            student_data=student_data,
            predicted_score=score,
            shap_explanation=attribution,
            student_name="Demo Student",
        )

        logger.info("\n" + "=" * 60)
        logger.info(f"DEMO FEEDBACK (backend: {gen.backend.name})")
        logger.info(f"Predicted score: {score:.1f}/100")
        logger.info("=" * 60)
        logger.info(feedback)

    except Exception as e:
        logger.warning(f"LLM feedback demo skipped: {e}")
        logger.warning("Set LLM_BACKEND in .env to enable. See .env.example.")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    results = main()
    sys.exit(0)
