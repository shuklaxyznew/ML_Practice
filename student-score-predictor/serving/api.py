"""
serving/api.py
--------------
FastAPI REST API for serving the trained student score prediction model.

Endpoints:
  POST /predict           single student prediction
  POST /predict/batch     batch predictions (more efficient than N single calls)
  POST /predict/explain   prediction + feature attribution + LLM feedback
  GET  /health            liveness check for load balancers / Kubernetes probes
  GET  /model/info        model metadata and feature contract

Production patterns implemented:
  - Pydantic validation with field bounds (fail before reaching the model)
  - Model + scaler loaded ONCE at startup via lifespan, not per-request
  - Structured logging with per-request UUIDs for distributed tracing
  - Proper HTTP error codes and informative messages
  - training=False on all inference calls (disables Dropout/BatchNorm train mode)
"""

import logging
import os
import pickle
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================================================
# Pydantic schemas — request / response contracts
# ======================================================================

class StudentFeatures(BaseModel):
    """
    Input schema with automatic validation.
    Pydantic raises 422 Unprocessable Entity before your code runs.
    """
    hours_studied_per_week:      float = Field(..., ge=0, le=60,  description="Weekly study hours (0-60)")
    attendance_percentage:       float = Field(..., ge=0, le=100, description="Attendance % (0-100)")
    assignments_completion_rate: float = Field(..., ge=0, le=100, description="Assignment completion % (0-100)")
    previous_exam_score:         float = Field(..., ge=0, le=100, description="Previous exam score (0-100)")
    sleep_hours_per_night:       float = Field(..., ge=0, le=12,  description="Sleep hours per night (0-12)")
    student_name:                Optional[str] = Field(None, description="Optional student name")

    model_config = {
        "json_schema_extra": {
            "example": {
                "hours_studied_per_week": 18.0,
                "attendance_percentage": 82.0,
                "assignments_completion_rate": 90.0,
                "previous_exam_score": 72.0,
                "sleep_hours_per_night": 7.5,
                "student_name": "Alex",
            }
        }
    }


class PredictionResponse(BaseModel):
    request_id:      str
    predicted_score: float
    confidence_band: dict    # ±1 MAE range
    risk_level:      str


class ExplainResponse(BaseModel):
    request_id:      str
    predicted_score: float
    risk_level:      str
    attribution:     dict    # feature → contribution value
    top_concern:     str
    top_strength:    str
    feedback:        Optional[str]


class BatchRequest(BaseModel):
    students: List[StudentFeatures]


# ======================================================================
# App state — loaded once, shared across all requests
# ======================================================================

class AppState:
    model:   Optional[tf.keras.Model] = None
    scaler:  Optional[object]         = None
    mae:     float                    = 5.0  # updated from training results
    version: str                      = "1.0.0"


state = AppState()

FEATURE_ORDER = [
    "hours_studied_per_week",
    "attendance_percentage",
    "assignments_completion_rate",
    "previous_exam_score",
    "sleep_hours_per_night",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: runs at startup and shutdown.
    Load model and scaler ONCE at startup — not per request.
    Per-request loading would be ~100-1000x slower.
    """
    model_path  = os.environ.get("MODEL_PATH",  "checkpoints/best_model")
    scaler_path = os.environ.get("SCALER_PATH", "artifacts/scaler.pkl")

    if os.path.exists(model_path):
        state.model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from: {model_path} ({state.model.count_params():,} params)")
    else:
        logger.warning(f"Model not found at '{model_path}'. Run train.py first.")
        # Load untrained model for demo purposes
        from model.architecture import build_model
        state.model = build_model()
        logger.warning("Using untrained model — predictions are meaningless until trained.")

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            state.scaler = pickle.load(f)
        logger.info(f"Scaler loaded from: {scaler_path}")
    else:
        logger.warning(f"Scaler not found at '{scaler_path}'. Predictions will be unscaled.")

    logger.info("API startup complete.")
    yield
    logger.info("API shutdown.")


# ======================================================================
# App
# ======================================================================

app = FastAPI(
    title="Student Score Prediction API",
    description=(
        "Predicts student final exam scores using a trained TensorFlow DNN. "
        "Supports single prediction, batch prediction, and LLM-generated feedback."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ======================================================================
# Endpoints
# ======================================================================

@app.get("/health", tags=["Infrastructure"])
async def health():
    """Liveness check for load balancers and Kubernetes readiness probes."""
    return {
        "status": "ok",
        "model_loaded":  state.model is not None,
        "scaler_loaded": state.scaler is not None,
        "version":       state.version,
    }


@app.get("/model/info", tags=["Infrastructure"])
async def model_info():
    """Model metadata — useful for auditing and API consumers."""
    if state.model is None:
        raise HTTPException(503, "Model not loaded. Run train.py first.")
    return {
        "model_name":     state.model.name,
        "version":        state.version,
        "parameters":     state.model.count_params(),
        "input_features": FEATURE_ORDER,
        "output":         "final_exam_score (0-100, continuous)",
        "training_mae":   state.mae,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(student: StudentFeatures):
    """
    Single student score prediction.

    Fast endpoint — no SHAP, no LLM. Use for real-time scoring.
    """
    rid = _request_id()
    X = _preprocess(student)
    score = _infer(X)
    logger.info(f"[{rid}] Predicted {score:.1f} for '{student.student_name or 'unknown'}'")

    return PredictionResponse(
        request_id=rid,
        predicted_score=round(score, 2),
        confidence_band={
            "low":  round(max(0.0, score - state.mae), 2),
            "high": round(min(100.0, score + state.mae), 2),
        },
        risk_level=_risk_level(score),
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """
    Batch prediction — more efficient than N individual POST /predict calls.

    All preprocessing and inference is done in one vectorised pass.
    """
    if not request.students:
        raise HTTPException(400, "students list is empty.")

    X_all = np.vstack([_preprocess(s) for s in request.students])
    scores = state.model(X_all, training=False).numpy().flatten()
    scores = np.clip(scores, 0, 100)

    results = [
        {
            "name":            s.student_name or f"student_{i}",
            "predicted_score": round(float(sc), 2),
            "risk_level":      _risk_level(float(sc)),
        }
        for i, (s, sc) in enumerate(zip(request.students, scores))
    ]
    return {"predictions": results, "count": len(results)}


@app.post("/predict/explain", response_model=ExplainResponse, tags=["Prediction"])
async def predict_explain(student: StudentFeatures):
    """
    Full pipeline: prediction → feature attribution → LLM coaching report.

    Slower than /predict due to:
      1. Gradient computation for feature attribution (~10ms)
      2. LLM API call / Ollama inference (~1-10s depending on backend)

    Use for counselor dashboards, not real-time scoring.
    """
    rid = _request_id()
    X = _preprocess(student)
    score = _infer(X)

    # Feature attribution (gradient-based — works without SHAP)
    from model.explainability import gradient_importance
    attribution = gradient_importance(state.model, X)

    # Sort by contribution
    sorted_attrs = sorted(attribution.items(), key=lambda kv: kv[1])
    top_concern  = sorted_attrs[0][0]  if sorted_attrs else "N/A"
    top_strength = sorted_attrs[-1][0] if sorted_attrs else "N/A"

    # LLM feedback (optional — gracefully skips if backend unavailable)
    feedback = None
    try:
        from genai.feedback_generator import StudentFeedbackGenerator
        gen = StudentFeedbackGenerator()
        raw_data = {
            k: getattr(student, k)
            for k in FEATURE_ORDER
        }
        feedback = gen.generate(
            student_data=raw_data,
            predicted_score=score,
            shap_explanation=attribution,
            student_name=student.student_name,
        )
    except Exception as exc:
        logger.warning(f"[{rid}] LLM feedback skipped: {exc}")

    logger.info(f"[{rid}] Explained prediction {score:.1f} for '{student.student_name or 'unknown'}'")

    return ExplainResponse(
        request_id=rid,
        predicted_score=round(score, 2),
        risk_level=_risk_level(score),
        attribution={k: round(v, 4) for k, v in attribution.items()},
        top_concern=top_concern,
        top_strength=top_strength,
        feedback=feedback,
    )


# ======================================================================
# Internal helpers
# ======================================================================

def _preprocess(student: StudentFeatures) -> np.ndarray:
    X = np.array([[
        student.hours_studied_per_week,
        student.attendance_percentage,
        student.assignments_completion_rate,
        student.previous_exam_score,
        student.sleep_hours_per_night,
    ]], dtype=np.float32)
    if state.scaler is not None:
        X = state.scaler.transform(X).astype(np.float32)
    return X


def _infer(X: np.ndarray) -> float:
    if state.model is None:
        raise HTTPException(503, "Model not loaded.")
    raw = float(state.model(X, training=False).numpy().flatten()[0])
    return float(np.clip(raw, 0, 100))


def _risk_level(score: float) -> str:
    if score >= 75:
        return "low"
    elif score >= 55:
        return "moderate"
    return "high"


def _request_id() -> str:
    return str(uuid.uuid4())[:8]


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api:app", host="0.0.0.0", port=8000, reload=True)
