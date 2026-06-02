# Student Score Prediction System

> A production-grade ML pipeline that predicts student exam scores, explains *why* using feature attribution, and generates personalized coaching feedback using a free local LLM (Ollama) or any free cloud API.

Built as a serious learning reference covering **custom TensorFlow training loops**, **explainable AI**, and **GenAI integration** — not a toy tutorial. Designed for engineers with strong software backgrounds entering the ML/GenAI space.

---

## Table of Contents

- [What This Project Actually Does](#what-this-project-actually-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Running the Project](#running-the-project)
- [Module Reference](#module-reference)
  - [data/generator.py](#datageneratorpy)
  - [data/pipeline.py](#datapipelinepy)
  - [model/architecture.py](#modelarchitecturepy)
  - [model/trainer.py](#modeltrainerpy)
  - [model/explainability.py](#modelexplainabilitypy)
  - [genai/feedback_generator.py](#genaifeedback_generatorpy)
  - [serving/api.py](#servingapipy)
  - [experiments/mlflow_tracking.py](#experimentsmlflow_trackingpy)
- [Key Concepts Explained](#key-concepts-explained)
  - [Regression vs Classification](#regression-vs-classification)
  - [Why MSE as Loss](#why-mse-as-loss)
  - [GradientTape vs fit()](#gradienttape-vs-fit)
  - [Feature Scaling and Data Leakage](#feature-scaling-and-data-leakage)
  - [What the LLM is Actually Doing](#what-the-llm-is-actually-doing)
  - [SHAP Values Explained](#shap-values-explained)
  - [Connection to LoRA and Transformers](#connection-to-lora-and-transformers)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [MLflow UI](#mlflow-ui)
- [LLM Backend Options](#llm-backend-options)
- [Extending This Project](#extending-this-project)
- [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)

---

## What This Project Actually Does

There are two completely separate ML components here. Understanding this distinction is critical:

```
Component 1: TensorFlow DNN (trained from scratch)
─────────────────────────────────────────────────────
  - Learns from 2000 synthetic student records
  - Runs a custom tf.GradientTape training loop
  - Outputs a predicted exam score (0-100)
  - THIS is where ML training happens

Component 2: LLM (pre-trained, just prompted)
─────────────────────────────────────────────────────
  - NOT trained at all — it is already trained by someone else
  - Used as a smart text formatter
  - Input  : ML prediction + feature attribution values
  - Output : Natural language coaching paragraph
  - THIS is just an API call with a structured prompt
```

The LLM component is called "Augmented Generation" — you ground a pre-trained LLM in your own data (the prediction and feature importance) before asking it to generate text. This is the same underlying pattern as RAG, just without retrieval.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Student Score Prediction System                    │
│                                                                      │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────────────┐ │
│  │  DATA LAYER  │    │   ML CORE      │    │   GenAI LAYER        │ │
│  │              │    │                │    │                      │ │
│  │ generator.py │───▶│ architecture.py│───▶│ feedback_generator   │ │
│  │              │    │                │    │                      │ │
│  │ Multivariate │    │ Functional API │    │ Pluggable backend:   │ │
│  │ normal dist  │    │ 3-layer DNN    │    │   Ollama (local)     │ │
│  │ Correlated   │    │                │    │   Groq (free cloud)  │ │
│  │ features     │    │ trainer.py     │    │   HuggingFace        │ │
│  │ Outlier group│    │                │    │   Gemini             │ │
│  │              │    │ GradientTape   │    │   Anthropic          │ │
│  │ pipeline.py  │    │ custom loop    │    │                      │ │
│  │              │    │                │    │ Score + attribution  │ │
│  │ StandardScale│    │ explainability │    │ → coaching report    │ │
│  │ Train/Val/   │    │                │    │                      │ │
│  │ Test split   │    │ Gradient attr  │    └──────────────────────┘ │
│  └──────────────┘    └────────────────┘              │               │
│         │                    │                       │               │
│         ▼                    ▼                       ▼               │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────────────┐ │
│  │  ARTIFACTS   │    │   SERVING      │    │   TRACKING           │ │
│  │              │    │                │    │                      │ │
│  │ scaler.pkl   │    │ FastAPI        │    │ MLflow               │ │
│  │ model ckpt   │    │ /predict       │    │ Hyperparams          │ │
│  │ dataset csv  │    │ /predict/batch │    │ Loss curves          │ │
│  │              │    │ /predict/expln │    │ Model artifacts      │ │
│  └──────────────┘    └────────────────┘    └──────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
student-score-predictor/
│
├── data/
│   ├── __init__.py
│   ├── generator.py          # Synthetic data with multivariate normal + correlations
│   └── pipeline.py           # StandardScaler, train/val/test split, scaler persistence
│
├── model/
│   ├── __init__.py
│   ├── architecture.py       # Functional API: Dense → BatchNorm → ReLU → Dropout
│   ├── trainer.py            # Custom tf.GradientTape loop: forward, backward, update
│   └── explainability.py     # SHAP (primary) + gradient attribution (fallback)
│
├── genai/
│   ├── __init__.py
│   └── feedback_generator.py # Pluggable LLM: Ollama / Groq / HuggingFace / Gemini / Anthropic
│
├── serving/
│   ├── __init__.py
│   └── api.py                # FastAPI: /predict, /predict/batch, /predict/explain
│
├── experiments/
│   ├── __init__.py
│   └── mlflow_tracking.py    # MLflow: hyperparams, metrics, model/scaler artifacts
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # 30+ unit + smoke tests (pytest)
│
├── notebooks/                # EDA, visualisations — add your own
├── checkpoints/              # Saved model checkpoints (git-ignored)
├── artifacts/                # Scaler pickles (git-ignored)
├── logs/                     # Training logs (git-ignored)
│
├── train.py                  # Main entry point — runs the full pipeline
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| ML Framework | TensorFlow 2.x | GradientTape is the foundation of LLM fine-tuning |
| Data | NumPy, Pandas, scikit-learn | Standard data science stack |
| Explainability | SHAP + gradient fallback | Gold standard for NN interpretability |
| GenAI | Pluggable (Ollama/Groq/HF/Gemini/Anthropic) | Free options available; same pattern across all |
| Serving | FastAPI + Uvicorn | Async, fast, auto-validated, OpenAPI docs included |
| Tracking | MLflow | Industry standard — compare runs, version artifacts |
| Testing | pytest | Clean, readable, covers all layers |

---

## Setup and Installation

### 1. Clone

```bash
git clone https://github.com/yourusername/student-score-predictor.git
cd student-score-predictor
```

### 2. Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
# venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env — set LLM_BACKEND and any required API keys
```

### 5. LLM Backend Setup

**Ollama (recommended — local, free, no key):**
```bash
# Install from https://ollama.ai, then:
ollama pull llama3.1        # 4.7 GB, best quality
# or
ollama pull gemma2:2b       # 1.6 GB, lightweight
```

**Groq (free cloud, fastest):**
```bash
# Sign up free at https://console.groq.com
# Add to .env:
# GROQ_API_KEY=gsk_...
# LLM_BACKEND=groq
```

**HuggingFace (free cloud, most model choice):**
```bash
# Sign up free at https://huggingface.co → Settings → Access Tokens
# Add to .env:
# HF_API_KEY=hf_...
# LLM_BACKEND=huggingface
```

**Gemini (Google free tier):**
```bash
# Get key at https://aistudio.google.com
# Add to .env:
# GEMINI_API_KEY=...
# LLM_BACKEND=gemini
```

---

## Running the Project

### Train the Model

```bash
# Default: 2000 samples, 100 epochs
python train.py

# Custom hyperparameters
python train.py --epochs 150 --batch_size 32 --lr 5e-4 --dropout 0.2

# Use an existing dataset
python train.py --data_path data/raw/student_data_20240601.csv

# Skip LLM feedback (runs without any LLM backend configured)
python train.py --no_feedback

# Skip MLflow logging
python train.py --no_mlflow
```

Training logs appear in both stdout and `logs/training.log`.

### Start the API Server

```bash
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs: `http://localhost:8000/docs`

### Run Tests

```bash
pytest tests/ -v
pytest tests/ -v --tb=short
pytest tests/ -v --cov=. --cov-report=html   # coverage report in htmlcov/
```

### Launch MLflow UI

```bash
mlflow ui --backend-store-uri experiments/mlruns
# Open http://localhost:5000
```

---

## Module Reference

### data/generator.py

Generates synthetic student data with realistic statistical structure.

**Key design decisions:**

The naive approach generates features independently:
```python
# Wrong — no real-world structure
hours = np.random.uniform(0, 60, 1000)
score = np.random.uniform(0, 100, 1000)
```

The correct approach uses a **multivariate normal distribution** with a covariance matrix that encodes domain knowledge (students who attend class also tend to submit assignments):

```python
corr = np.array([
    [1.00, 0.40, 0.35, 0.30, 0.05],  # hours_studied
    [0.40, 1.00, 0.60, 0.25, 0.05],  # attendance
    [0.35, 0.60, 1.00, 0.20, 0.05],  # assignments
    [0.30, 0.25, 0.20, 1.00, 0.10],  # previous_score
    [0.05, 0.05, 0.05, 0.10, 1.00],  # sleep
])
D = np.diag(stds)
cov = D @ corr @ D  # correlation → covariance: Σ = D·R·D
```

**Score formula** — causal, not arbitrary:
```
score = 0.30 × previous_score
      + 0.25 × hours_studied
      + 0.20 × attendance
      + 0.15 × assignments
      + 0.10 × sleep
      + N(0, 5)     ← irreducible noise
```

The noise term represents things no model can know: illness on exam day, a difficult paper, personal events.

**Outlier group (5%):** High study hours but low scores. Simulates burnout or test anxiety. Without this, the model learns a spuriously clean relationship between effort and score.

---

### data/pipeline.py

Preprocessing: load → validate → scale → split.

**The data leakage trap** — common production bug:

```python
# WRONG — scaler sees test distribution during fit
scaler.fit(X_all)
X_train, X_test = split(scaler.transform(X_all))

# CORRECT — scaler sees only training distribution
X_train, X_test = split(X_all)
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)   # apply training stats
```

If you fit the scaler on all data including test, the model implicitly knows the test set's mean and variance during training. Evaluation metrics look better than production performance. The pipeline enforces the correct order.

The scaler is saved as a timestamped `.pkl` artifact. At serving time, the API loads this exact pickle — not a fresh unfitted StandardScaler.

---

### model/architecture.py

TensorFlow model built with the **Functional API** (not Sequential).

```
Input (5 features)
    │
    ▼
Dense(128, He init, L2 reg, no bias) → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(64,  He init, L2 reg, no bias) → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(32,  He init, L2 reg, no bias) → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(1, linear)    ← unbounded regression output
```

**Why Functional API over Sequential:**
Sequential hides the computation graph. Functional API exposes intermediate tensors, which is required for SHAP, multi-output models, attention mechanisms, and Transformer architectures. HuggingFace models are all built this way.

**Why He uniform initialisation:**
For ReLU activations, He initialisation sets initial weights from a distribution with variance = 2 / fan_in. This preserves the signal variance across layers on the forward pass. Glorot (Xavier) initialisation is designed for tanh and will cause ReLU networks to lose signal in deep layers.

**Why linear output for regression:**
Sigmoid caps outputs at [0, 1]. Softmax forces outputs to sum to 1. Regression requires the output to be unbounded — linear activation passes the pre-activation value directly through.

---

### model/trainer.py

Custom training loop using `tf.GradientTape`. This is the most important file in the project for your LoRA and fine-tuning learning path.

**The GradientTape pattern — memorise this:**

```python
with tf.GradientTape() as tape:
    y_pred = model(X_batch, training=True)     # forward pass
    loss   = loss_fn(y_batch, y_pred)          # MSE
    loss  += sum(model.losses)                 # L2 regularisation

# tape has recorded all operations above
# .gradient() traverses the computation graph backward
# computing ∂loss/∂w for every trainable weight
gradients = tape.gradient(loss, model.trainable_variables)

# Clip gradient vector if global L2 norm exceeds threshold
# Prevents a single bad batch destroying weights (critical for Transformers)
gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

# Update: w = w - lr × ∂loss/∂w
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

**Why `@tf.function` on the step methods:**
`@tf.function` compiles the Python function into a TensorFlow computation graph on the first call. Subsequent calls execute the compiled graph directly, bypassing Python overhead. Typically 2-10x faster than eager execution for training loops.

**Early stopping implementation:**
The trainer tracks `best_val_loss` and a `patience_counter`. When counter reaches `patience` epochs without improvement, training stops. On exit, it restores the weights from the best checkpoint — not the weights from the final epoch. This is a common mistake: training for 100 epochs, taking the final weights, but the model actually peaked at epoch 73.

---

### model/explainability.py

Feature attribution: answers "why did the model predict this score?"

**Two modes:**

1. **SHAP DeepExplainer** (primary): Uses a background dataset to compute expected values, then applies DeepLIFT to approximate Shapley values. Theoretically grounded, satisfies consistency and completeness axioms. Requires `pip install shap`.

2. **Gradient attribution** (fallback): Computes ∂output/∂input averaged over samples. Much faster, zero extra dependencies, less precise but sufficient for the LLM prompt context.

**SHAP value interpretation:**

```
Average score (base value): 72.0
Predicted score:            54.0

Feature attribution:
  sleep_hours_per_night:        -9.8   (4hrs sleep — severe impact)
  previous_exam_score:          -6.4   (scored 45 last time)
  attendance_percentage:        -4.1   (missed 30% of classes)
  hours_studied_per_week:       +2.3   (studying well)
  assignments_completion_rate:  +0.0   (neutral)
  ──────────────────────────────────
  Sum:                          -18.0  → 72.0 - 18.0 = 54.0 ✓
```

SHAP values always sum to (prediction - base_value). This completeness property is what makes them trustworthy for explainability audits.

---

### genai/feedback_generator.py

Converts ML output (score + feature attribution) into natural language coaching.

**What it is and is not:**

The LLM is not being trained. It is a pre-trained model being called with a structured prompt. The "intelligence" in the feedback comes from the LLM's pre-training, guided by:
1. A system prompt defining the counselor role and constraints
2. A structured user prompt containing the ML model's findings

**The prompt construction** is the critical engineering here:

```
STUDENT PERFORMANCE ANALYSIS
Predicted Score: 54.0/100
Risk: high risk — urgent intervention needed

CURRENT METRICS:
  Weekly study hours : 12.0 hrs
  Class attendance   : 65.0%
  ...

FACTORS HURTING THIS STUDENT:
  - Sleep Hours (impact: -9.8 pts)
  - Class Attendance (impact: -6.1 pts)
  ...
```

This structured context grounds the LLM's response in the actual model findings, rather than generating generic advice.

**Backend abstraction:** All five backends implement `LLMBackend.complete(system, user) -> str`. Swapping backends requires only changing `LLM_BACKEND` in `.env`.

---

### serving/api.py

FastAPI REST API with three prediction endpoints.

**Startup pattern — load once, not per-request:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup
    state.model  = tf.keras.models.load_model(model_path)
    state.scaler = pickle.load(open(scaler_path, "rb"))
    yield
    # Runs at shutdown
```

Loading a Keras model takes ~1-2 seconds. If you loaded it per-request, your API would have ~2s latency on every call and would not scale.

**Pydantic validation** fires before any application code runs. A request with `attendance_percentage=150` returns HTTP 422 with a structured error response — never reaches the model.

**The three endpoints:**

| Endpoint | Speed | When to use |
|----------|-------|-------------|
| `POST /predict` | Fast (~5ms) | Real-time scoring, dashboards |
| `POST /predict/batch` | Fast, vectorised | Processing a cohort |
| `POST /predict/explain` | Slow (~2-10s) | Counselor tools, one-off analysis |

The `explain` endpoint is slower because it runs gradient attribution and makes an LLM API call. Never use it in a hot path.

---

### experiments/mlflow_tracking.py

Experiment tracking via MLflow context manager.

```python
with ExperimentTracker() as tracker:
    tracker.log_config(config)          # hyperparameters
    tracker.log_history(history)        # per-epoch metrics (time series)
    tracker.log_test_results(results)   # final evaluation
    tracker.log_model(model)            # model artifact
    tracker.log_scaler(scaler_path)     # scaler artifact (same run)
```

**What gets stored per run:**
- All hyperparameters (epochs, lr, batch_size, dropout, patience)
- Per-epoch: train_loss, val_loss, train_mae, val_mae, RMSE, learning_rate
- Summary: best_val_loss, best_epoch
- Test set: MSE, MAE, RMSE, R²
- Model artifact (saved TF model)
- Scaler artifact (must be version-locked with model)
- Tags: model_type, framework, data_type

**Why scaler and model must be in the same run:** A model checkpoint trained with scaler A will produce wrong predictions if served with scaler B. By logging both in the same MLflow run, you can always retrieve the matched pair.

---

## Key Concepts Explained

### Regression vs Classification

This project predicts a **continuous score (0-100)**, not a category.

Classification would require binning scores (A/B/C/F), which discards information. A classifier predicting "B grade" cannot tell you whether the student scored 74 or 84. Regression preserves the full numerical signal, allows the model to express uncertainty as a range, and enables more nuanced intervention decisions.

Rule: continuous numerical output → regression; discrete category output → classification.

### Why MSE as Loss

Mean Squared Error = mean((predicted − actual)²)

Squaring serves two purposes:
1. Makes all errors positive (no cancellation between over- and under-predictions)
2. Penalizes large errors quadratically — being off by 20 points is penalized 4× more than being off by 10

MSE is differentiable everywhere, which matters for gradient-based optimisation.

We track MAE as a monitoring metric because it is in the same unit as the label (score points) and is easier to explain to non-technical stakeholders: "on average, predictions are off by X points."

R² (coefficient of determination) measures how much variance the model explains. R² = 0.87 means the model explains 87% of the score variance; the remaining 13% is irreducible noise (illness on exam day, paper difficulty, etc.).

### GradientTape vs fit()

`model.fit()` is a convenience wrapper. Under the hood it does exactly what the GradientTape loop does, but it hides everything.

| Capability | `.fit()` | `GradientTape` |
|------------|---------|---------------|
| Access gradients mid-step | No | Yes |
| Custom loss per layer | Hard | Trivial |
| Freeze specific weights | Limited | Full control |
| Gradient clipping | Via callback | Direct |
| LoRA-style partial updates | Very hard | Natural |
| Multi-task losses | Complex | Straightforward |

When you study LoRA fine-tuning, the training script will look almost identical to `model/trainer.py`. The only difference is that `model.trainable_variables` contains only the injected low-rank matrices, not the frozen base weights. The tape pattern is identical.

### Feature Scaling and Data Leakage

StandardScaler transforms each feature to zero mean and unit variance using statistics computed from the data. The critical rule:

**Compute statistics from training data only. Apply them to all splits.**

```python
# Correct order — enforced in data/pipeline.py
X_train, X_test = split(X)
scaler.fit(X_train)              # statistics from train only
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)   # apply train statistics to test
```

If you fit the scaler before splitting (the common mistake), the test set's distribution statistics contaminate the scaler. The model sees information about the test set during training. Evaluation metrics look better than they will be in production.

This is called **data leakage** and is one of the most common bugs in ML pipelines.

### What the LLM is Actually Doing

The LLM in this project is used for **inference only** — it is not trained, not fine-tuned, not updated in any way.

Think of it as calling a very sophisticated function:

```python
llm.complete(
    system="You are an academic counselor...",
    user=f"Student predicted to score {score:.0f}/100. Sleep is their biggest issue..."
)
# → Returns: "Alex, I can see you're working hard, but I'm concerned about..."
```

The LLM's knowledge comes entirely from its pre-training. Your job as an ML engineer is to write a prompt that grounds the LLM's response in your model's findings, so the output is specific to this student rather than generic advice.

This is the same principle as Retrieval Augmented Generation (RAG), just without the retrieval step.

### SHAP Values Explained

SHAP is rooted in Shapley values from cooperative game theory. The intuition:

Imagine the prediction is a game, and the features are players. Run the model many times with different subsets of players present. For each feature, average its marginal contribution across all possible orderings of the other players. That average is the SHAP value.

**Properties that matter in production:**
- **Completeness**: SHAP values sum to (prediction - base_value). The math always reconciles.
- **Consistency**: If a feature always increases the output, its SHAP value is always positive.
- **Model-agnostic**: Works on neural networks, trees, or any model.

These properties are why SHAP is trusted for regulatory explainability (GDPR, EU AI Act) while simpler techniques like permutation importance are not.

### Connection to LoRA and Transformers

Everything in this project maps directly to LLM fine-tuning:

| This project | LoRA fine-tuning |
|-------------|-----------------|
| `tf.GradientTape` loop | Identical pattern |
| `model.trainable_variables` | Only LoRA rank matrices (base frozen) |
| Gradient clipping (norm=1.0) | Standard in all Transformer training |
| CosineDecay LR schedule | With linear warmup in Transformers |
| Early stopping | Same concept |
| Prompt engineering in `_build_prompt` | Instruction dataset design |
| SHAP attribution | Attention visualization, mechanistic interpretability |

LoRA is not magic. It is a GradientTape loop where:
1. Base model weights are frozen (`layer.trainable = False`)
2. Small low-rank matrices are injected and marked trainable
3. The same training loop runs, but `tape.gradient()` only receives the LoRA parameters

You already understand step 3 from this project.

---

## API Reference

### POST /predict

Single score prediction. Fast, no LLM.

**Request:**
```json
{
  "hours_studied_per_week": 18.0,
  "attendance_percentage": 82.0,
  "assignments_completion_rate": 90.0,
  "previous_exam_score": 72.0,
  "sleep_hours_per_night": 7.5,
  "student_name": "Alex"
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

### POST /predict/batch

Vectorised batch prediction.

**Request:**
```json
{
  "students": [
    { "hours_studied_per_week": 18.0, "attendance_percentage": 82.0, ... },
    { "hours_studied_per_week": 8.0,  "attendance_percentage": 55.0, ... }
  ]
}
```

### POST /predict/explain

Full pipeline: prediction + attribution + LLM feedback.

**Response:**
```json
{
  "request_id": "d9e1f4a8",
  "predicted_score": 52.3,
  "risk_level": "moderate",
  "attribution": {
    "Sleep Hours": 0.28,
    "Previous Exam Score": 0.26,
    "Class Attendance": 0.22,
    "Weekly Study Hours": 0.15,
    "Assignment Completion": 0.09
  },
  "top_concern": "Sleep Hours",
  "top_strength": "Weekly Study Hours",
  "feedback": "Alex, it's clear you're putting in real effort with your studies..."
}
```

### GET /health

```json
{ "status": "ok", "model_loaded": true, "scaler_loaded": true, "version": "1.0.0" }
```

### GET /model/info

```json
{
  "model_name": "StudentScorePredictor",
  "parameters": 18177,
  "input_features": ["hours_studied_per_week", "attendance_percentage", ...],
  "output": "final_exam_score (0-100, continuous)"
}
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific class
pytest tests/test_pipeline.py::TestStudentDataGenerator -v
pytest tests/test_pipeline.py::TestTrainer -v

# With coverage
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Stop on first failure
pytest tests/ -x -v
```

**Test coverage:**

| Class | Tests | What is verified |
|-------|-------|-----------------|
| `TestStudentDataGenerator` | 9 | Shape, ranges, nulls, reproducibility, correlations, outlier logic |
| `TestDataPipeline` | 8 | Split ratios, leakage, dtype, reproducibility, scaler stats |
| `TestModelArchitecture` | 7 | Shape, compilation, params, naming, determinism, gradients |
| `TestTrainer` | 4 | Smoke run, finite loss, metric keys, early stopping |
| `TestExplainability` | 3 | Keys, finite values, normalisation |

---

## MLflow UI

```bash
mlflow ui --backend-store-uri experiments/mlruns
# Open http://localhost:5000
```

The UI shows:
- All training runs with full hyperparameter tables
- Per-epoch metric charts (hover to see exact values)
- Run comparison: select multiple rows → Compare
- Artifact browser: click any run → Artifacts to see saved model and scaler

To find your best run: sort the Runs table by `test.r2` descending.

---

## LLM Backend Options

| Backend | Cost | Speed | Quality | Setup |
|---------|------|-------|---------|-------|
| Ollama (llama3.1) | Free | ~3-8s | Excellent | Install app, pull model |
| Ollama (gemma2:2b) | Free | ~1-2s | Good | Install app, pull model |
| Groq (llama-3.1-70b) | Free tier | ~0.5s | Excellent | API key |
| HuggingFace (mistral-7b) | Free tier | ~5-30s | Good | API key |
| Gemini 1.5 Flash | Free tier | ~1s | Good | API key |
| Anthropic Claude | Paid | ~1-2s | Best | API key |

Switch by setting `LLM_BACKEND` in `.env`. No code changes needed.

---

## Extending This Project

**Add real data:**
Replace `data/generator.py` with a loader for UCI Student Performance dataset or any real source. The pipeline, model, and API require zero changes — only the generator changes.

**Hyperparameter search:**
Add `optuna` or `keras_tuner`. Log each trial to MLflow automatically. Compare dozens of configurations visually in the MLflow UI.

**Multi-task learning:**
Add a second output head predicting `pass_fail` (binary classification) alongside the regression score. The Functional API already supports this. Requires a combined loss in the GradientTape loop — a natural next step.

**Fine-tune a Transformer:**
Replace the DNN with a small Transformer encoder (e.g., DistilBERT for tabular embeddings). The GradientTape loop in `model/trainer.py` requires **zero changes** — only the model architecture changes. This is the direct bridge to your LoRA study.

**LoRA adapter:**
```python
# Freeze base Dense layers
for layer in model.layers:
    if "dense" in layer.name:
        layer.trainable = False

# Add small trainable adapters (low-rank matrices)
# Run the same GradientTape loop — only adapter weights update
```

**Drift detection:**
Add a monitoring endpoint that computes KL-divergence or Population Stability Index between incoming request feature distributions and the training distribution. Alert when features drift beyond a threshold.

---

## Design Decisions and Trade-offs

| Decision | Chosen | Considered | Reason |
|----------|--------|-----------|--------|
| Data generation | Multivariate normal + covariance | Independent uniform | Real correlations matter for learning |
| Model API | Functional | Sequential | Extensible; exposes computation graph |
| Training loop | GradientTape | `.fit()` | LoRA relevance; full gradient access |
| Explainability | SHAP + gradient fallback | Permutation importance | SHAP is theoretically grounded |
| LLM integration | Pluggable backends | Hardcoded single API | Free options; swappable without code change |
| Serving | FastAPI | Flask | Async; Pydantic validation; auto OpenAPI |
| Tracking | MLflow | Weights & Biases | Self-hosted; open source; no account required |
| Scaler | StandardScaler | MinMaxScaler, RobustScaler | Zero-mean unit-variance is optimal for neural nets; RobustScaler preferable with many outliers |
| Gradient clipping | Global norm = 1.0 | Per-parameter | Global norm is the standard in Transformer training |

---

## License

MIT

---

## Author

Built as a production-grade learning reference for engineers with strong software backgrounds entering the ML and GenAI space.

Topics covered: TensorFlow Functional API, custom GradientTape training, StandardScaler data leakage, SHAP explainability, Augmented Generation with LLMs, FastAPI serving patterns, MLflow experiment tracking, and the direct connection from all of the above to LoRA fine-tuning.
