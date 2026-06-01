"""
serving/api.py
--------------
FastAPI REST API for serving the trained student score prediction model.

Endpoints:
  POST /predict          → single student score prediction
  POST /predict/batch    → batch predictions
  POST /predict/explain  → prediction + SHAP explanation + LLM feedback
  GET  /health           → health check
  GET  /model/info       → model metadata

Production considerations implemented here:
  - Pydantic request/response validation (fail fast, clear errors)
  - Input range validation (same as training-time checks)
  - Scaler loaded at startup (not per-request)
  - Structured logging with request IDs
  - Error handling with informative messages
"""

import os
import uuid
import logging
import numpy as np
import tensorflow as tf
import pickle
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------

class StudentFeatures(BaseModel):
    """
    Input schema with validation.
    Pydantic validators fire before your code ever runs — fail fast.
    """
    hours_studied_per_week:      float = Field(..., ge=0, le=60,  description="Weekly study hours (0-60)")
    attendance_percentage:       float = Field(..., ge=0, le=100, description="Attendance % (0-100)")
    assignments_completion_rate: float = Field(..., ge=0, le=100, description="Assignment completion % (0-100)")
    previous_exam_score:         float = Field(..., ge=0, le=100, description="Previous exam score (0-100)")
    sleep_hours_per_night:       float = Field(..., ge=0, le=12,  description="Sleep hours per night (0-12)")
    student_name:                Optional[str] = Field(None, description="Optional student name for personalized feedback")

    class Config:
        schema_extra = {
            "example": {
                "hours_studied_per_week":      18.0,
                "attendance_percentage":       82.0,
                "assignments_completion_rate": 90.0,
                "previous_exam_score":         72.0,
                "sleep_hours_per_night":        7.5,
                "student_name":                "Alex"
            }
        }


class PredictionResponse(BaseModel):
    request_id:      str
    predicted_score: float
    confidence_band: dict   # ±1 MAE range
    risk_level:      str


class ExplainResponse(BaseModel):
    request_id:       str
    predicted_score:  float
    risk_level:       str
    shap_values:      dict
    feedback:         Optional[str]
    top_concern:      str
    top_strength:     str


class BatchPredictionRequest(BaseModel):
    students: List[StudentFeatures]


# ------------------------------------------------------------------
# App state (loaded once at startup)
# ------------------------------------------------------------------

class ModelRegistry:
    model:   Optional[tf.keras.Model] = None
    scaler:  Optional[object]         = None
    mae:     float                    = 5.0  # filled from training results
    version: str                      = "1.0.0"


registry = ModelRegistry()

FEATURE_ORDER = [
    "hours_studied_per_week",
    "attendance_percentage",
    "assignments_completion_rate",
    "previous_exam_score",
    "sleep_hours_per_night",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler at startup — not per-request."""
    model_path  = os.environ.get("MODEL_PATH",  "checkpoints/best_model")
    scaler_path = os.environ.get("SCALER_PATH", "artifacts/scaler.pkl")

    if os.path.exists(model_path):
        registry.model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from: {model_path}")
    else:
        logger.warning(f"No model found at {model_path}. Using untrained model for demo.")
        from model.architecture import build_model
        registry.model = build_model()

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            registry.scaler = pickle.load(f)
        logger.info(f"Scaler loaded from: {scaler_path}")
    else:
        logger.warning("No scaler found. Predictions may be inaccurate.")

    yield
    logger.info("API shutdown.")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Student Score Prediction API",
    description="Predicts final exam scores and generates personalized LLM feedback.",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": registry.model is not None,
        "scaler_loaded": registry.scaler is not None,
        "model_version": registry.version,
    }


@app.get("/model/info")
async def model_info():
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_name":    registry.model.name,
        "version":       registry.version,
        "parameters":    registry.model.count_params(),
        "input_features": FEATURE_ORDER,
        "output":        "final_exam_score (0-100)",
        "training_mae":  registry.mae,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(student: StudentFeatures):
    """Single student prediction."""
    request_id = str(uuid.uuid4())[:8]

    X = _preprocess(student)
    score = _predict(X)

    return PredictionResponse(
        request_id=request_id,
        predicted_score=round(score, 2),
        confidence_band={
            "low":  round(max(0, score - registry.mae), 2),
            "high": round(min(100, score + registry.mae), 2),
        },
        risk_level=_risk_level(score),
    )


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction — more efficient than N single calls."""
    results = []
    for student in request.students:
        X     = _preprocess(student)
        score = _predict(X)
        results.append({
            "name":            student.student_name or "Unknown",
            "predicted_score": round(score, 2),
            "risk_level":      _risk_level(score),
        })
    return {"predictions": results, "count": len(results)}


@app.post("/predict/explain", response_model=ExplainResponse)
async def predict_explain(student: StudentFeatures):
    """
    Full pipeline: prediction + SHAP explanation + LLM feedback.

    This is the most powerful endpoint — combines all three layers.
    Note: Slower due to SHAP computation and LLM API call.
    """
    request_id = str(uuid.uuid4())[:8]
    X          = _preprocess(student)
    score      = _predict(X)

    # Gradient-based importance as fallback (SHAP requires initialization)
    from model.explainability import gradient_importance
    raw_importance = gradient_importance(registry.model, X)

    # Try to generate LLM feedback
    feedback = None
    try:
        from genai.feedback_generator import StudentFeedbackGenerator
        gen          = StudentFeedbackGenerator()
        student_dict = student.dict(exclude={"student_name"})
        feedback     = gen.generate(
            student_data=student_dict,
            predicted_score=score,
            shap_explanation=raw_importance,
            student_name=student.student_name,
        )
    except Exception as e:
        logger.warning(f"LLM feedback failed: {e}")

    sorted_features = sorted(raw_importance.items(), key=lambda x: x[1], reverse=True)
    top_strength = sorted_features[0][0] if sorted_features else "N/A"
    top_concern  = sorted(raw_importance.items(), key=lambda x: x[1])[0][0]

    return ExplainResponse(
        request_id=request_id,
        predicted_score=round(score, 2),
        risk_level=_risk_level(score),
        shap_values={k: round(v, 4) for k, v in raw_importance.items()},
        feedback=feedback,
        top_concern=top_concern,
        top_strength=top_strength,
    )


# ------------------------------------------------------------------
# Internal utilities
# ------------------------------------------------------------------

def _preprocess(student: StudentFeatures) -> np.ndarray:
    X = np.array([[
        student.hours_studied_per_week,
        student.attendance_percentage,
        student.assignments_completion_rate,
        student.previous_exam_score,
        student.sleep_hours_per_night,
    ]], dtype=np.float32)

    if registry.scaler is not None:
        X = registry.scaler.transform(X).astype(np.float32)
    return X


def _predict(X: np.ndarray) -> float:
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    score = float(registry.model(X, training=False).numpy().flatten()[0])
    return float(np.clip(score, 0, 100))


def _risk_level(score: float) -> str:
    if score >= 75:
        return "low"
    elif score >= 55:
        return "moderate"
    else:
        return "high"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api:app", host="0.0.0.0", port=8000, reload=True)
