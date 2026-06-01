# Student Score Prediction System

> A production-grade ML pipeline that predicts student exam scores, explains *why* using SHAP, and generates personalized coaching feedback using a Large Language Model.

Built as a serious learning project covering **custom TensorFlow training loops**, **explainable AI**, and **GenAI integration** — not a toy tutorial.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [What You Will Learn](#what-you-will-learn)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Module Deep Dives](#module-deep-dives)
  - [Data Pipeline](#1-data-pipeline)
  - [Model Architecture](#2-model-architecture)
  - [Custom Training Loop](#3-custom-training-loop--tf-gradienttape)
  - [SHAP Explainability](#4-shap-explainability)
  - [GenAI Feedback Layer](#5-genai-feedback-layer)
  - [FastAPI Serving](#6-fastapi-serving)
  - [MLflow Experiment Tracking](#7-mlflow-experiment-tracking)
- [Key ML Concepts Explained](#key-ml-concepts-explained)
  - [Why Regression, Not Classification](#why-regression-not-classification)
  - [MSE vs MAE — When to Use Which](#mse-vs-mae--when-to-use-which)
  - [Why GradientTape Over .fit()](#why-gradienttape-over-fit)
  - [Feature Scaling — The Leakage Trap](#feature-scaling--the-leakage-trap)
  - [SHAP Values — Intuition](#shap-values--intuition)
  - [Connection to LoRA & Transformers](#connection-to-lora--transformers)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [MLflow UI](#mlflow-ui)
- [Extending This Project](#extending-this-project)
- [Design Decisions & Trade-offs](#design-decisions--trade-offs)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Student Score Prediction System                    │
│                                                                      │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────────────┐ │
│  │  DATA LAYER  │    │   ML CORE      │    │   GenAI LAYER        │ │
│  │              │    │                │    │                      │ │
│  │ generator.py │───▶│ architecture.py│───▶│ feedback_generator   │ │
│  │              │    │                │    │                      │ │
│  │ Multivariate │    │ Functional API │    │ Claude (Anthropic)   │ │
│  │ normal dist  │    │ 3-layer DNN    │    │                      │ │
│  │ Correlated   │    │                │    │ Prediction + SHAP    │ │
│  │ features     │    │ trainer.py     │    │ → Natural language   │ │
│  │ Outlier group│    │                │    │   coaching report    │ │
│  │              │    │ GradientTape   │    │                      │ │
│  │ pipeline.py  │    │ custom loop    │    └──────────────────────┘ │
│  │              │    │                │              │               │
│  │ StandardScale│    │ explainability │              │               │
│  │ Train/Val/   │    │                │              │               │
│  │ Test split   │    │ SHAP values    │──────────────┘               │
│  └──────────────┘    └────────────────┘                              │
│         │                    │                                       │
│         ▼                    ▼                                       │
│  ┌──────────────┐    ┌────────────────┐                             │
│  │  ARTIFACTS   │    │   SERVING      │                             │
│  │              │    │                │                             │
│  │ scaler.pkl   │    │ FastAPI        │                             │
│  │ model ckpt   │    │ /predict       │                             │
│  │ dataset csv  │    │ /predict/batch │                             │
│  │              │    │ /predict/explain│                            │
│  └──────────────┘    └────────────────┘                             │
│                               │                                      │
│                               ▼                                      │
│                      ┌────────────────┐                             │
│                      │   TRACKING     │                             │
│                      │                │                             │
│                      │ MLflow         │                             │
│                      │ Hyperparams    │                             │
│                      │ Metrics        │                             │
│                      │ Artifacts      │                             │
│                      └────────────────┘                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What You Will Learn

This project is structured as a **learning ladder** — each module introduces one core concept and builds on the previous one.

| Module | Concept | Why It Matters |
|--------|---------|----------------|
| `data/generator.py` | Multivariate normal distributions, correlated feature generation | Real data isn't independent. Naive `np.random.uniform` doesn't teach your model realistic patterns. |
| `data/pipeline.py` | StandardScaler, train/val/test split, data leakage prevention | Fitting the scaler on test data is one of the most common production bugs. |
| `model/architecture.py` | Functional API, BatchNorm, Dropout, He initialization | Sequential API hides too much. Functional API is how Transformers are built. |
| `model/trainer.py` | `tf.GradientTape`, gradient clipping, early stopping | `.fit()` is a black box. Fine-tuning LLMs requires this exact pattern. |
| `model/explainability.py` | SHAP values, feature attribution | EU AI Act, GDPR require explainable predictions. Also essential for debugging. |
| `genai/feedback_generator.py` | LLM prompt engineering, structured data → natural language | Bridges ML outputs with GenAI. Foundation of augmented generation. |
| `serving/api.py` | FastAPI, Pydantic validation, model serving | A model that isn't served is a research project, not a product. |
| `experiments/mlflow_tracking.py` | Experiment tracking, reproducibility | Without tracking, you'll forget what you tried and why it worked. |

---

## Project Structure

```
student-score-predictor/
│
├── data/
│   ├── __init__.py
│   ├── generator.py          # Synthetic data with realistic distributions + correlations
│   └── pipeline.py           # Feature scaling, splitting, scaler persistence
│
├── model/
│   ├── __init__.py
│   ├── architecture.py       # Functional API model (not Sequential)
│   ├── trainer.py            # Custom tf.GradientTape training loop
│   └── explainability.py     # SHAP + gradient-based feature attribution
│
├── genai/
│   ├── __init__.py
│   └── feedback_generator.py # Claude API → personalized student reports
│
├── serving/
│   ├── __init__.py
│   └── api.py                # FastAPI endpoints: predict, batch, explain
│
├── experiments/
│   ├── __init__.py
│   └── mlflow_tracking.py    # Run logging, metric tracking, artifact storage
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Unit + smoke tests (pytest)
│
├── notebooks/                # EDA, visualization (add your own)
├── checkpoints/              # Model checkpoints (git-ignored)
├── artifacts/                # Scaler pickles, exports (git-ignored)
├── logs/                     # Training logs (git-ignored)
│
├── train.py                  # Main training orchestrator (entry point)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| ML Framework | TensorFlow 2.x | Industry standard; GradientTape is the foundation of LLM fine-tuning |
| Data | NumPy, Pandas, Scikit-learn | Standard data science stack |
| Explainability | SHAP | Gold standard for model interpretability |
| GenAI | Anthropic Claude API | Converts structured ML output to natural language |
| Serving | FastAPI + Uvicorn | Async, fast, Pydantic-validated API |
| Experiment Tracking | MLflow | Industry standard for ML experiment management |
| Testing | pytest | Clean, readable test suite |

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/student-score-predictor.git
cd student-score-predictor
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

```env
# .env
ANTHROPIC_API_KEY=sk-ant-...
MODEL_PATH=checkpoints/best_model
SCALER_PATH=artifacts/scaler.pkl
```

> **Note:** The LLM feedback layer (`/predict/explain`) requires an Anthropic API key.
> All other features (training, prediction, SHAP) work without it.

---

## Running the Project

### Train the Model

```bash
# Default: generate 2000 samples, train 100 epochs
python train.py

# Custom configuration
python train.py --epochs 150 --batch_size 32 --lr 5e-4 --dropout 0.2

# Use existing dataset
python train.py --data_path data/raw/student_data_20240101.csv

# Skip LLM feedback demo (no API key needed)
python train.py --no_feedback
```

Training logs appear in `logs/training.log` and stdout simultaneously.

### Start the API Server

```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs available at: `http://localhost:8000/docs`

### Run Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=html   # with coverage
```

### Launch MLflow UI

```bash
mlflow ui --backend-store-uri experiments/mlruns
# Open http://localhost:5000
```

---

## Module Deep Dives

### 1. Data Pipeline

**File:** `data/generator.py`

The key insight here is that real student data is **not independently distributed**. Students who attend class regularly also tend to complete assignments. Generating features independently with `np.random.uniform` creates a dataset that doesn't resemble reality — and your model will learn wrong patterns.

**Solution:** Multivariate normal distribution with a covariance matrix.

```python
# Correlation matrix encodes domain knowledge:
# attendance ↔ assignments: 0.6 (strong positive)
# hours_studied ↔ previous_score: 0.3 (mild positive)
corr = np.array([
    [1.00, 0.40, 0.35, 0.30, 0.05],  # hours_studied
    [0.40, 1.00, 0.60, 0.25, 0.05],  # attendance
    ...
])
samples = np.random.multivariate_normal(means, cov, size=n)
```

**Outlier group (5%):** Simulates burnout students — high study hours but low scores. This tests model robustness and prevents the model from learning a naive "more study = always higher score" rule.

**Score formula** (weighted sum + Gaussian noise):
```
score = 0.30 × previous_score
      + 0.25 × hours_studied
      + 0.20 × attendance
      + 0.15 × assignments
      + 0.10 × sleep
      + N(0, 5)   ← irreducible noise
```

The noise term is critical — it represents things no model can know: student felt sick that day, exam was unusually hard, etc.

---

### 2. Model Architecture

**File:** `model/architecture.py`

Uses the **Functional API**, not Sequential. This matters because:

- Sequential hides the computation graph
- Functional API exposes intermediate tensors (needed for SHAP, attention maps, etc.)
- This is exactly how Transformer models are structured in HuggingFace

```
Input (5 features)
    │
    ▼
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(32)  → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(1, activation=linear)   ← regression output
```

**Key decisions:**

| Decision | Choice | Reason |
|----------|--------|--------|
| Activation | ReLU | Simple, no vanishing gradient, works well for tabular data |
| Initialization | He uniform | Mathematically optimal for ReLU — preserves variance across layers |
| BatchNorm position | Before activation | Stabilizes input distribution to each activation |
| Output activation | Linear | Regression output must be unbounded (0-100 range) |
| Optimizer | Adam | Adaptive learning rates — works well without extensive LR tuning |

---

### 3. Custom Training Loop — `tf.GradientTape`

**File:** `model/trainer.py`

This is the most important module for your LoRA/fine-tuning journey.

The `GradientTape` pattern is used in:
- HuggingFace `Trainer` class
- LoRA / QLoRA fine-tuning scripts
- Any custom multi-task or multi-loss setup

```python
with tf.GradientTape() as tape:
    y_pred = model(X_batch, training=True)    # forward pass
    loss   = loss_fn(y_batch, y_pred)         # compute MSE
    loss  += sum(model.losses)                # add L2 regularization

# Compute ∂loss/∂weights for every trainable parameter
gradients = tape.gradient(loss, model.trainable_variables)

# Clip to prevent exploding gradients (critical in deep nets & Transformers)
gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

# Apply: w = w - lr × gradient
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

**Why gradient clipping?** In deep networks (and especially Transformers), gradients can grow exponentially through layers during backprop. Without clipping, a single bad batch can destroy your model's weights. `clip_by_global_norm` rescales the entire gradient vector if its L2 norm exceeds the threshold.

**Connection to LoRA:** In LoRA fine-tuning, you freeze the base model weights and only train the injected low-rank matrices. This is done by setting `trainable=False` on frozen layers — and passing only the LoRA parameters to `tape.gradient()`. Same `GradientTape` pattern, just selective weight updates.

---

### 4. SHAP Explainability

**File:** `model/explainability.py`

SHAP (SHapley Additive exPlanations) answers: **"Why did the model predict this score?"**

**Intuition:** Imagine all features playing a game together to produce the prediction. SHAP assigns each feature its fair share of "credit" (or blame) for the prediction, based on all possible orderings of features.

**Example output for a student predicted to score 52:**

```
Predicted Score: 52.0/100
Base value (average): 72.0

+ Weekly Study Hours:      +3.1  (studied 18hrs — above average)
+ Assignment Completion:   +0.3  (submitted most work)
- Previous Exam Score:     -6.4  (scored 45 last time)
- Class Attendance:        -8.2  (attended only 55%)
- Sleep Hours:             -9.8  (averaging 4hrs — severe deficit)
```

This output feeds directly into the LLM prompt — grounding the generated feedback in actual model evidence.

**Two modes implemented:**
1. `ModelExplainer` (SHAP DeepExplainer) — precise, requires SHAP library
2. `gradient_importance` — gradient-based fallback, no extra dependencies

---

### 5. GenAI Feedback Layer

**File:** `genai/feedback_generator.py`

This demonstrates the **Augmented Generation** pattern — not RAG (no retrieval), but the same principle: ground the LLM in structured data before asking it to generate.

**Flow:**
```
ML prediction (score=52.0)
    +
SHAP values (attendance=-8.2, sleep=-9.8, ...)
    │
    ▼
Structured prompt (not just "write feedback")
    │
    ▼
Claude API (claude-sonnet-4-20250514)
    │
    ▼
Personalized coaching report in natural language
```

**System prompt design principles applied:**
- Role assignment: "experienced academic counselor"
- Constraint: do NOT mention ML/SHAP/model (ruins trust)
- Tone: warm but actionable
- Length: bounded (150-250 words)
- Format: plain paragraphs (not bullets — feels more human)

This is your first step toward understanding prompt engineering at a systems level — the same principles apply to tool-use prompts, chain-of-thought prompts, and fine-tuning instruction datasets.

---

### 6. FastAPI Serving

**File:** `serving/api.py`

Three endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Single prediction — fast, no LLM |
| `/predict/batch` | POST | Batch predictions — efficient |
| `/predict/explain` | POST | Full pipeline: prediction + SHAP + LLM report |
| `/health` | GET | Liveness check for load balancers |
| `/model/info` | GET | Model metadata |

**Production patterns implemented:**
- Pydantic validation with field bounds (fail before touching the model)
- Model loaded once at startup via `lifespan` context (not per-request)
- Scaler loaded once at startup — serving-time consistency with training
- Structured error responses
- Request IDs for tracing

---

### 7. MLflow Experiment Tracking

**File:** `experiments/mlflow_tracking.py`

Without experiment tracking, you'll run 50 experiments and forget what worked and why.

**What gets logged per run:**
- All hyperparameters (epochs, lr, dropout, batch size)
- Per-epoch: train_loss, val_loss, train_mae, val_mae, RMSE, LR
- Test set: MSE, MAE, RMSE, R²
- Model artifact (saved model)
- Scaler artifact (for reproducibility)
- Tags: model type, dataset version, framework

```bash
mlflow ui --backend-store-uri experiments/mlruns
```

The UI shows you loss curves, metric comparisons across runs, and all logged artifacts.

---

## Key ML Concepts Explained

### Why Regression, Not Classification

This model predicts a **continuous score (0–100)**, not a category.

Classification would require you to bin scores (e.g., A/B/C/F) — which throws away information. A model predicting "B grade" can't tell you if the student scored 74 or 84. Regression preserves the full numerical signal.

Rule of thumb:
- Output is a number on a continuous scale → Regression
- Output is one of N discrete categories → Classification

### MSE vs MAE — When to Use Which

| Metric | Formula | Behavior |
|--------|---------|----------|
| MSE | mean((ŷ - y)²) | Squares errors — large errors penalized quadratically |
| MAE | mean(\|ŷ - y\|) | Linear penalty — robust to outliers |
| RMSE | √MSE | Same unit as label — directly interpretable |

**This project uses MSE as the loss** because:
- Squaring makes it differentiable everywhere (important for gradient descent)
- Penalizing large errors more is desirable here — a prediction off by 20 points is worse than 4× as bad as one off by 10

**We track MAE as a metric** because it's easier to explain: "on average, predictions are off by X score points."

**R² (coefficient of determination):** Tells you how much variance your model explains. R²=0.85 means the model explains 85% of the variance in scores — the remaining 15% is irreducible noise or missing features.

### Why GradientTape Over `.fit()`

`.fit()` is fine for prototyping. But it abstracts away everything:

| `.fit()` | `GradientTape` |
|----------|---------------|
| Automatic | Manual — you control every step |
| Fixed loss per model | Custom loss per batch, per layer, per step |
| Can't inspect gradients | Full gradient access |
| Hard to freeze/unfreeze layers | Trivial: filter `trainable_variables` |
| LoRA? Very hard | LoRA is exactly this pattern |

When you study LoRA, you'll see: freeze base weights → inject trainable rank matrices → run this exact loop on only those matrices. The `GradientTape` muscle memory you build here transfers directly.

### Feature Scaling — The Leakage Trap

**The problem:**

```python
# WRONG — data leakage
scaler.fit_transform(X)          # fit on ALL data including test
X_train, X_test = split(X)       # too late, test stats contaminated the scaler
```

**The correct order:**

```python
# CORRECT — no leakage
X_train, X_test = split(X)       # split first
scaler.fit(X_train)              # fit ONLY on train
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)  # apply train stats to test
```

Why it matters: If you fit the scaler on test data, the model implicitly "sees" test distribution statistics during training. Evaluation metrics look better than they'll be in production. This is one of the most common and subtle bugs in ML pipelines.

### SHAP Values — Intuition

SHAP is rooted in cooperative game theory (Shapley values from economics).

**Simple intuition:** Run the model thousands of times with different subsets of features. For each feature, measure: "how much does including this feature change the prediction?" Average this over all possible orderings. That average contribution is the SHAP value.

**Properties that make SHAP trustworthy:**
- **Consistency:** If a feature always increases the output, its SHAP value is positive
- **Completeness:** SHAP values sum to the difference between the prediction and the base rate
- **Model-agnostic:** Works on any model (neural nets, trees, linear models)

### Connection to LoRA & Transformers

Here's how everything you're learning connects to your GenAI study track:

```
This project                    LoRA / Transformer fine-tuning
─────────────────────────       ────────────────────────────────────
GradientTape loop          →    Fine-tuning loop
Freeze layers (BatchNorm)  →    Freeze base model weights
Selective weight update    →    Update only LoRA rank matrices
Gradient clipping (1.0)    →    Standard in Transformer training
Learning rate scheduling   →    Warmup + cosine decay (same concept)
Prompt engineering         →    Instruction tuning dataset design
SHAP attribution           →    Attention visualization / mechanistic interp
```

LoRA is not magic — it's a `GradientTape` loop where you inject small trainable matrices and only backpropagate into those. Building this intuition on a simple regression model, where the math is transparent, is the right foundation.

---

## API Reference

### POST `/predict`

Predict a single student's score.

**Request:**
```json
{
  "hours_studied_per_week": 18.0,
  "attendance_percentage": 82.0,
  "assignments_completion_rate": 90.0,
  "previous_exam_score": 72.0,
  "sleep_hours_per_night": 7.5
}
```

**Response:**
```json
{
  "request_id": "a3f7b2c1",
  "predicted_score": 76.4,
  "confidence_band": { "low": 71.4, "high": 81.4 },
  "risk_level": "low"
}
```

### POST `/predict/explain`

Full pipeline: prediction + feature importance + LLM coaching report.

**Response:**
```json
{
  "request_id": "d9e1f4a8",
  "predicted_score": 52.3,
  "risk_level": "moderate",
  "shap_values": {
    "Weekly Study Hours": 3.1,
    "Class Attendance": -8.2,
    "Assignment Completion": 0.3,
    "Previous Exam Score": -6.4,
    "Sleep Hours": -9.8
  },
  "feedback": "Alex, it's clear you're putting in real effort with your studies...",
  "top_concern": "Sleep Hours",
  "top_strength": "Weekly Study Hours"
}
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test class
pytest tests/test_pipeline.py::TestStudentDataGenerator -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html

# Stop on first failure
pytest tests/ -x
```

**Test categories:**

| Class | What it tests |
|-------|--------------|
| `TestStudentDataGenerator` | Shape, dtypes, ranges, reproducibility, correlations |
| `TestDataPipeline` | Split ratios, data leakage, dtype consistency |
| `TestModelArchitecture` | Output shape, parameter count, gradient flow |
| `TestTrainer` | End-to-end smoke test, metric keys, finite values |

---

## MLflow UI

```bash
mlflow ui --backend-store-uri experiments/mlruns
```

Open `http://localhost:5000`

You'll see:
- All training runs with full hyperparameters
- Loss curves per epoch (train vs val)
- Metric comparisons across runs
- Saved model artifacts
- Scaler artifacts for reproducibility

To compare runs: check multiple rows → "Compare" button.

---

## Extending This Project

This project is intentionally structured for extension. Here are natural next steps:

### Add Real Data
Replace `data/generator.py` with a loader for a real dataset (UCI Student Performance, Kaggle, etc.). The pipeline, model, and serving layer require zero changes.

### Hyperparameter Tuning
Add `keras_tuner` or `optuna` to search over `hidden_units`, `dropout_rate`, `learning_rate`. Log each trial to MLflow.

### Multi-Task Learning
Add a second output head: predict `pass_fail` (classification) alongside `score` (regression). This requires the Functional API (already used here) and a custom combined loss in the `GradientTape` loop.

### Fine-Tune a Transformer
Replace the DNN with a small Transformer encoder. The `GradientTape` training loop in `trainer.py` requires **zero changes** — only the model changes. This is the direct bridge to your LoRA study.

### LoRA Layer
Implement a simple LoRA-style adapter:
```python
# Freeze base Dense layers
for layer in model.layers:
    layer.trainable = False

# Add trainable low-rank adapters
# Train with GradientTape — same loop, different trainable_variables
```

### Drift Detection
Add a monitoring endpoint that computes feature distribution statistics on incoming requests and alerts when they drift from the training distribution.

---

## Design Decisions & Trade-offs

| Decision | What was chosen | What was considered | Why |
|----------|----------------|--------------------|----|
| Data generation | Multivariate normal + covariance | Independent uniform sampling | Realistic correlations matter for learning |
| Model API | Functional | Sequential | Functional is extensible; Sequential hides computation graph |
| Training loop | GradientTape | `.fit()` | Direct LoRA/fine-tuning relevance; full gradient access |
| Explainability | SHAP + gradient fallback | Permutation importance | SHAP is theoretically grounded; gradient is zero-dependency |
| LLM | Claude (Anthropic) | GPT-4, Gemini | Structured outputs, reliable instruction following |
| Serving | FastAPI | Flask, Django | Async, Pydantic validation, automatic OpenAPI docs |
| Tracking | MLflow | Weights & Biases | Self-hosted, open source, no account required |
| Scaler | StandardScaler | MinMaxScaler, RobustScaler | Neural nets work well with zero-mean unit-variance inputs; RobustScaler is better if you have many outliers |

---

## License

MIT

---

## Author

Built as a production-grade learning project covering TensorFlow custom training, explainable AI, and GenAI integration. Designed for engineers with strong software backgrounds entering the ML/GenAI space.
