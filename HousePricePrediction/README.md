# 🏠 House Price Prediction using TensorFlow

A complete end-to-end Machine Learning project that predicts California house prices using a deep neural network built with TensorFlow and Keras. This project was built step-by-step covering every core ML concept — from raw data to a saved, deployable model.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [ML Concepts Covered](#ml-concepts-covered)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [What Happens When You Run It](#what-happens-when-you-run-it)
- [Output Files Generated](#output-files-generated)
- [How to Use the Saved Model for Testing](#how-to-use-the-saved-model-for-testing)
- [Key Line Numbers Reference](#key-line-numbers-reference)
- [Expected Results](#expected-results)
- [Plots Explained](#plots-explained)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This project predicts the **median house value** (in units of $100,000) for housing districts in California based on features like income, location, house age, and population density.

The entire pipeline is covered in one file:

```
Raw Data → EDA → Feature Engineering → Scaling → Neural Network → Evaluation → Save Model
```

This was built as a learning project. Every function, every decision, and every line of code has a comment explaining **what** it does and **why** it was done that way.

---

## Dataset

**Name:** California Housing Dataset  
**Source:** Built into `scikit-learn` — no download needed, loads automatically  
**Loaded at:** Line 70 in `house_price_prediction.py`

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()   # Line 70
```

### Dataset Details

| Property | Value |
|---|---|
| Total Samples | 20,640 rows |
| Original Features | 8 columns |
| Engineered Features | 3 additional columns (created in code) |
| Final Feature Count | 11 columns |
| Target Column | `MedHouseVal` (Median House Value × $100k) |
| Missing Values | None |
| Source | 1990 California Census |

### Original Feature Descriptions

| Feature | Description | Example Value |
|---|---|---|
| `MedInc` | Median income of households in the block | 3.5 = $35,000/year |
| `HouseAge` | Median age of houses in the block (years) | 25 |
| `AveRooms` | Average number of rooms per household | 5.2 |
| `AveBedrms` | Average number of bedrooms per household | 1.1 |
| `Population` | Total population of the block | 1200 |
| `AveOccup` | Average number of occupants per household | 2.8 |
| `Latitude` | Geographic latitude of the block | 37.88 |
| `Longitude` | Geographic longitude of the block | -122.23 |

### Engineered Features (created at Lines 94–96)

| Feature | Formula | Why Created |
|---|---|---|
| `rooms_per_household` | `AveRooms / AveOccup` | Raw room count is less useful than rooms per person |
| `bedrooms_per_room` | `AveBedrms / AveRooms` | Bedroom ratio indicates housing type |
| `population_per_household` | `Population / AveOccup` | Measures household density in the block |

### Target Variable

| Property | Value |
|---|---|
| Column | `MedHouseVal` |
| Unit | $100,000 (so 3.5 = $350,000) |
| Min | $14,999 |
| Max | $500,001 (artificially capped — important limitation) |
| Note | The dataset caps all values above $500k at exactly $500k. This creates a visible horizontal band in plots at y=5.0 |

---

## Model Architecture

**Model Type:** Feedforward Neural Network (Multi-Layer Perceptron)  
**Framework:** TensorFlow 2.x / Keras Sequential API  
**Task:** Regression (predicting a continuous value)

**Model is built at:** Lines 409–438 in `house_price_prediction.py`  
**Model is compiled at:** Lines 444–448  
**Model is saved at:** Line 790

### Layer-by-Layer Breakdown

```
Input Layer
  └── 11 features (8 original + 3 engineered, all StandardScaled)

Hidden Layer 1  [Line 413–418]
  └── Dense(64 neurons)
  └── Activation: ReLU  → max(0, x), avoids vanishing gradients
  └── Regularizer: L2(0.001) → penalizes large weights
  └── Dropout(0.3)  [Line 421] → randomly disables 30% of neurons

Hidden Layer 2  [Line 424–428]
  └── Dense(32 neurons)
  └── Activation: ReLU
  └── Regularizer: L2(0.001)
  └── Dropout(0.3)  [Line 429]

Hidden Layer 3  [Line 432]
  └── Dense(16 neurons)
  └── Activation: ReLU

Output Layer  [Line 437]
  └── Dense(1 neuron)
  └── NO activation function
      → We want a raw real number, not a probability
```

### Compilation Settings

| Setting | Value | Why |
|---|---|---|
| Optimizer | `adam` | Adapts learning rate per parameter automatically |
| Loss | `mse` | Mean Squared Error — standard for regression |
| Metric | `mae` | Mean Absolute Error — interpretable in $100k units |

### Regularization (prevents overfitting)

| Technique | Setting | Line | Purpose |
|---|---|---|---|
| Dropout | 0.3 (30%) | 421, 429 | Randomly disables neurons — forces redundant learning |
| L2 Weight Decay | 0.001 | 416, 427 | Penalizes large weights — keeps model generalized |
| Early Stopping | patience=10 | 491–496 | Stops training when val_loss stops improving |

### Training Configuration

| Setting | Value | Line |
|---|---|---|
| Max Epochs | 200 | 509 |
| Batch Size | 32 | 510 |
| Validation Split | 20% | 511 |
| Early Stopping Monitor | `val_loss` | 492 |
| Early Stopping Patience | 10 epochs | 493 |
| Restore Best Weights | `True` | 494 |

---

## ML Concepts Covered

| Step | Concept | Lines |
|---|---|---|
| Step 1 | Data loading, feature engineering, log1p transformation | 52–117 |
| Step 2 | EDA — histograms, correlation heatmap, scatter plots | 134–225 |
| Step 3 | Visualizing skew before/after log transformation | 238–284 |
| Step 4 | Train/test split (80/20), StandardScaler, data leakage prevention | 304–365 |
| Step 5 | Neural network build — Dense, ReLU, Dropout, L2, output layer | 390–454 |
| Step 6 | Training, EarlyStopping, plotting training curves | 475–605 |
| Step 7 | Evaluation — MSE, MAE, R², Actual vs Predicted, Residuals | 628–756 |
| Step 8 | Saving model (.keras) and scaler (.pkl) | 775–799 |
| Step 9 | Loading artifacts, running inference on new data | 814–847 |

---

## Project Structure

After running the script, your folder will look like this:

```
house-price-prediction/
│
├── house_price_prediction.py     ← Main script (the only file you need to run)
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
│
├── house_price_model.keras       ← Saved trained model  [created at Line 790]
├── scaler.pkl                    ← Saved StandardScaler [created at Line 795]
│
├── plot_01_feature_distributions.png   ← Histogram of all features
├── plot_02_correlation_heatmap.png     ← Feature correlation matrix
├── plot_03_medinc_vs_target.png        ← Income vs House Price scatter
├── plot_04_log_transform.png           ← Before/after log transformation
├── plot_05_training_history.png        ← Loss and MAE training curves
└── plot_06_evaluation.png              ← Actual vs Predicted + Residuals
```

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- ~500MB disk space (for TensorFlow installation)
- No GPU required — runs on CPU

To check your Python version:
```bash
python --version
```

---

## Installation

### Step 1 — Clone or Download the Repository

```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction
```

Or if you downloaded the ZIP:
```bash
cd house-price-prediction
```

### Step 2 — Create a Virtual Environment (Recommended)

A virtual environment keeps your project dependencies separate from your global Python.

**On macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt. This means the virtual environment is active.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs TensorFlow, scikit-learn, pandas, numpy, matplotlib, seaborn, and joblib.

Installation typically takes 2–5 minutes depending on your internet speed.

To verify TensorFlow installed correctly:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## How to Run

### Run the Full Pipeline

```bash
python house_price_prediction.py
```

That's it. One command runs everything — data loading, EDA, feature engineering, training, evaluation, saving, and inference demo.

### Expected Runtime
- On CPU (no GPU): approximately **2–5 minutes**
- EarlyStopping will stop training early (usually around epoch 50–100)
- 6 plot windows will open sequentially — **close each one to continue**

---

## What Happens When You Run It

When you run `python house_price_prediction.py`, the following steps execute in order. Watch your terminal — each step prints its progress.

### Step 1 — Data Loading (Line 63–117)
```
============================================================
STEP 1: Loading Data & Feature Engineering
============================================================
[INFO] Raw Dataset Loaded
  Shape          : (20640, 9)
  Features       : ['MedInc', 'HouseAge', ...]
  Target         : MedHouseVal (median house value x $100k)
  Missing Values : 0
[INFO] Basic Statistics: ...
[INFO] Interaction Features Added
[INFO] Log1p Transformation Applied
[INFO] Final Dataset Shape: (20640, 12)
```

### Step 2 — EDA Plots (Lines 144–225)
Three plot windows open one at a time. Close each to proceed.
- Plot 1: Histogram grid of all features
- Plot 2: Correlation heatmap
- Plot 3: Income vs House Price scatter

### Step 3 — Log Transform Visualization (Lines 244–284)
One plot window showing before/after distributions. Close to proceed.

### Step 4 — Train/Test Split (Lines 315–363)
```
[INFO] After Train/Test Split (80/20):
  X_train : (16512, 11)
  X_test  : (4128, 11)
[INFO] First 3 Features BEFORE Scaling:  [large numbers]
[INFO] First 3 Features AFTER  Scaling:  [mean ≈ 0, std ≈ 1]
[CHECK] Mean ≈ 0.0 and Std ≈ 1.0 confirms scaling worked.
```

### Step 5 — Model Summary (Lines 405–452)
```
Model: "sequential"
_________________________________________________________________
Layer (type)          Output Shape         Param #
=================================================================
dense (Dense)         (None, 64)           768
dropout (Dropout)     (None, 64)           0
dense_1 (Dense)       (None, 32)           2080
...
Total params: 3,393
```

### Step 6 — Training (Lines 498–603)
```
[TRAINING STARTED]
Epoch 1/200 - loss: 1.2345 - mae: 0.8901 - val_loss: 0.9876
Epoch 2/200 - loss: 0.8765 - mae: 0.7654 - val_loss: 0.8123
...
Epoch 00073: early stopping
[TRAINING COMPLETE]
  Trained for : 73 epochs
  Best epoch  : 63
```

### Step 7 — Evaluation (Lines 640–754)
```
[RESULTS] Test Set Metrics:
  MSE  : 0.3012  (mean squared error, lower = better)
  MAE  : 0.3456  (avg error = ~$35k per house)
  R²   : 0.8123  (model explains 81.2% of price variance)

[INTERPRETATION] Good model performance
```

### Step 8 — Save Artifacts (Lines 785–799)
```
[SAVED] house_price_model.keras
[SAVED] scaler.pkl
```

### Step 9 — Inference Demo (Lines 824–845)
```
[LOADED] house_price_model.keras
[LOADED] scaler.pkl

[DEMO] Prediction on First Test Sample:
  Predicted House Value : $213.4k
  Actual House Value    : $227.0k
  Absolute Error        : $13.6k
```

---

## Output Files Generated

### `house_price_model.keras` — The Trained Model
- **Created at:** Line 790
- **Contains:** Full neural network architecture, trained weights, optimizer state
- **Size:** ~100–200 KB
- **Format:** TensorFlow's native Keras format
- **Use for:** Loading and predicting without retraining

### `scaler.pkl` — The Feature Scaler
- **Created at:** Line 795
- **Contains:** The mean and standard deviation of each of the 11 training features
- **Size:** ~2 KB
- **Format:** Python joblib serialized object
- **Use for:** Transforming raw input before passing to the model
- **Critical:** You MUST use this exact scaler on any new input. A new scaler fitted on different data will produce wrong predictions.

---

## How to Use the Saved Model for Testing

After running the main script once, `house_price_model.keras` and `scaler.pkl` will exist in your folder. You can load and use them without retraining.

### Option 1 — Quick Test on One Sample (copy-paste ready)

Create a new file called `test_model.py` and paste this:

```python
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ── Load saved artifacts ──────────────────────────────────
model  = load_model("house_price_model.keras")
scaler = joblib.load("scaler.pkl")

# ── Define a raw input sample ────────────────────────────
# Feature order MUST match training:
# [MedInc, HouseAge, AveRooms, AveBedrms, Population,
#  AveOccup, Latitude, Longitude,
#  rooms_per_household, bedrooms_per_room, population_per_household]

# NOTE: AveRooms, AveBedrms, Population, AveOccup must be
#       log1p-transformed BEFORE passing to the scaler,
#       because that's how they were preprocessed during training.

import numpy as np

# Raw values (as they would come from a real district)
MedInc      = 3.5       # Median income $35,000
HouseAge    = 20        # Houses are ~20 years old
AveRooms    = 5.0       # 5 rooms on average
AveBedrms   = 1.1       # 1.1 bedrooms on average
Population  = 1200      # 1200 people in the block
AveOccup    = 2.8       # 2.8 occupants per household
Latitude    = 37.85     # Northern California
Longitude   = -122.25

# Apply the same log1p transformation used in training
AveRooms_log   = np.log1p(AveRooms)
AveBedrms_log  = np.log1p(AveBedrms)
Population_log = np.log1p(Population)
AveOccup_log   = np.log1p(AveOccup)

# Create engineered features
rooms_per_household      = AveRooms / AveOccup
bedrooms_per_room        = AveBedrms / AveRooms
population_per_household = Population / AveOccup

# Assemble the feature vector in the EXACT same order as training
sample = np.array([[
    MedInc,
    HouseAge,
    AveRooms_log,
    AveBedrms_log,
    Population_log,
    AveOccup_log,
    Latitude,
    Longitude,
    rooms_per_household,
    bedrooms_per_room,
    population_per_household
]])

# ── Scale using the saved scaler ─────────────────────────
sample_scaled = scaler.transform(sample)

# ── Predict ──────────────────────────────────────────────
prediction = model.predict(sample_scaled, verbose=0)[0][0]

print(f"Predicted House Value: ${prediction * 100:.1f}k")
```

Run it:
```bash
python test_model.py
```

### Option 2 — Test on Multiple Samples from the Dataset

```python
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Load artifacts
model  = load_model("house_price_model.keras")
scaler = joblib.load("scaler.pkl")

# Recreate the same test set (random_state=42 ensures same split)
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target

# Recreate engineered features
df["rooms_per_household"]      = df["AveRooms"] / df["AveOccup"]
df["bedrooms_per_room"]        = df["AveBedrms"] / df["AveRooms"]
df["population_per_household"] = df["Population"] / df["AveOccup"]

# Apply log transforms
for col in ["AveRooms", "AveBedrms", "Population", "AveOccup"]:
    df[col] = np.log1p(df[col])

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale using the saved scaler
X_test_scaled = scaler.transform(X_test)

# Predict on first 10 test samples
predictions = model.predict(X_test_scaled[:10], verbose=0).flatten()

print(f"{'Sample':<8} {'Actual ($k)':<16} {'Predicted ($k)':<16} {'Error ($k)'}")
print("-" * 55)
for i, (actual, pred) in enumerate(zip(y_test.values[:10], predictions)):
    print(f"{i+1:<8} ${actual*100:<14.1f} ${pred*100:<14.1f} ${abs(actual-pred)*100:.1f}")
```

---

## Key Line Numbers Reference

| What | Line Number |
|---|---|
| Dataset is loaded | 70 |
| Interaction features created | 94–96 |
| Log1p transformation applied | 111–112 |
| Train/test split | 328–332 |
| Scaler fitted on training data | 352 |
| Neural network definition starts | 409 |
| Layer 1 — Dense(64) + L2 | 413–418 |
| Dropout after Layer 1 | 421 |
| Layer 2 — Dense(32) + L2 | 424–428 |
| Dropout after Layer 2 | 429 |
| Layer 3 — Dense(16) | 432 |
| Output Layer — Dense(1), no activation | 437 |
| Model compiled | 444–448 |
| EarlyStopping defined | 491–496 |
| Training starts | 506 |
| **Model file created** (`house_price_model.keras`) | **790** |
| **Scaler file created** (`scaler.pkl`) | **795** |
| Model loaded from disk | 829 |
| Demo prediction | 839 |

---

## Expected Results

Results will vary slightly across runs due to random weight initialization, but typical values are:

| Metric | Typical Value | Meaning |
|---|---|---|
| Test MSE | 0.28 – 0.35 | Mean Squared Error on test set |
| Test MAE | 0.33 – 0.38 | Average error of ~$33k–$38k per house |
| Test R² | 0.78 – 0.83 | Model explains 78–83% of price variance |
| Epochs trained | 50 – 100 | EarlyStopping halts before 200 |

---

## Plots Explained

| File | What It Shows | What to Look For |
|---|---|---|
| `plot_01_feature_distributions.png` | Histogram of every feature | Right-skewed columns = need log transform |
| `plot_02_correlation_heatmap.png` | Pairwise correlations | `MedInc` ↔ `MedHouseVal` ≈ 0.69 (strongest) |
| `plot_03_medinc_vs_target.png` | Income vs House Price | Upward trend + hard ceiling at 5.0 |
| `plot_04_log_transform.png` | Before/after log1p on 4 features | Blue (before) = skewed; Red (after) = bell-shaped |
| `plot_05_training_history.png` | Loss and MAE per epoch | Both curves should converge (healthy training) |
| `plot_06_evaluation.png` | Actual vs Predicted + Residuals | Points near diagonal = good; residuals centered at 0 = unbiased |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**
```bash
pip install tensorflow
```

**`ModuleNotFoundError: No module named 'sklearn'`**
```bash
pip install scikit-learn
```

**Plots not showing (running on a server/headless environment)**

Add this near the top of the script, before any matplotlib imports:
```python
import matplotlib
matplotlib.use('Agg')  # Save plots without displaying them
```

**`FileNotFoundError: house_price_model.keras`**

You must run `house_price_prediction.py` fully first before using `test_model.py`. The model file is created at Line 790.

**Training is very slow**

This is normal on CPU. The model should complete in 2–5 minutes. If it's taking much longer, reduce epochs to 50 as a test:
```python
# In train_model(), change:
epochs=200  →  epochs=50
```

**`WARNING: All log messages before absl::InitializeLog() is called are written to STDERR`**

This is a normal TensorFlow startup warning. It does not affect results.

---

## Author

Built as a step-by-step ML learning project, covering the full pipeline from data exploration to model deployment.

---

## License

MIT License — free to use, modify, and share.
