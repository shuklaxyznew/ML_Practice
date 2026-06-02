# 📉 Linear Regression From Scratch

> **Building machine learning fundamentals without shortcuts.**  
> No scikit-learn models. No TensorFlow. Just Python, NumPy, Pandas, and your brain.

---

## 🎯 What This Project Is

This project teaches you **how machine learning really works** by building a complete Linear Regression model from zero — implementing every formula, every update step, and every metric by hand.

By the end, you'll understand:
- Why machine learning models need training
- How the model "learns" from data
- What gradient descent is and why it works
- How to measure if a model is any good

---

## 📁 Project Structure

```
linear_regression_from_scratch/
│
├── src/
│   ├── linear_regression.py   ← The model itself (hypothesis, cost, gradient descent)
│   ├── metrics.py             ← MSE, RMSE, MAE, R² — implemented manually
│   └── preprocessing.py       ← Data loading, normalisation, train/test split
│
├── outputs/                   ← Generated charts (auto-created on run)
├── main.py                    ← Run this to train and visualise everything
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone the project
git clone https://github.com/yourname/linear-regression-from-scratch
cd linear-regression-from-scratch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run everything
python main.py
```

That's it. The script generates synthetic data, trains two models, evaluates them, and saves charts to `./outputs/`.

---

## 🧠 Core Concepts Explained

### What is Linear Regression?

Linear regression finds the best straight line through your data. Given some features (inputs), it predicts a target (output).

```
predicted_salary = (years_experience × weight) + bias
```

The model's job is to find the `weight` and `bias` that make its predictions as close to reality as possible.

---

### The Hypothesis Function

```
ŷ = X · w + b
```

- `X` = input features (e.g., years of experience)
- `w` = learned weights (how much each feature matters)
- `b` = bias/intercept (baseline prediction)
- `ŷ` = prediction

For house prices with 3 features:
```
price = (size × w₁) + (bedrooms × w₂) + (age × w₃) + b
```

---

### The Cost Function (MSE)

```
MSE = (1/n) × Σ(ŷᵢ - yᵢ)²
```

MSE measures "how wrong is the model on average?" Lower is better. Squaring the errors:
- Makes all errors positive (no cancelling)
- Penalises large mistakes more than small ones
- Creates a smooth curve that's easy to optimise

---

### 🏔️ Gradient Descent — The Mountain Analogy

> *Imagine a blindfolded person standing somewhere on a mountain, trying to reach the lowest point in the valley. They can't see the whole landscape — but they can feel the slope under their feet. So they take one small step downhill. Then another. And another. Eventually, they reach the bottom — the point where the error is as small as possible.*

**This is Gradient Descent.**

- Your position on the mountain = the model's current weights
- The altitude = the current error (MSE)
- Each step downhill = one parameter update

**The update rule:**
```
w = w - learning_rate × (∂MSE/∂w)
b = b - learning_rate × (∂MSE/∂b)
```

The gradient (∂MSE/∂w) tells us the slope — which direction is "uphill." We subtract it to go downhill.

---

### 🎛️ Learning Rate

The learning rate controls how big each step is.

| Learning Rate | Effect |
|---|---|
| Too small (0.001) | Training is painfully slow — thousands of steps to converge |
| Too large (1.0) | Overshoots the valley — bounces around, may never converge |
| Just right (0.1) | Reaches the bottom efficiently |

Think of it as: small steps = cautious but slow. Large steps = fast but risky.

---

### Epochs

One **epoch** = one full pass through the entire training dataset.

We repeat many epochs because one pass isn't enough for the model to converge. With each pass, the weights inch closer to their optimal values.

---

### Convergence

The model has **converged** when the cost (error) stops decreasing meaningfully between epochs. You'll see this in the cost-vs-epochs chart: the line flattens out.

---

### Overfitting vs Underfitting

```
Underfitting: Model is too simple → high error on training AND test data
               (a straight line through a curved dataset)

Overfitting:  Model memorises training data → low training error, high test error
               (follows noise, not the real pattern)

Good fit:     Low error on both training and test data
```

The train/test split is how we detect overfitting — if training metrics are much better than test metrics, the model is overfitting.

---

### Bias and Weights

- **Weights (w):** How much each input feature contributes to the prediction.  
  A weight of 5000 on `YearsExperience` means "each extra year adds $5000 to salary."

- **Bias (b):** The baseline prediction when all features are zero.  
  Think of it as the y-intercept in the equation y = mx + b.

---

## 📊 Evaluation Metrics

### Mean Squared Error (MSE)
```
MSE = (1/n) × Σ(ŷᵢ - yᵢ)²
```
- Units: squared (e.g., dollars²)
- Lower is better
- Useful during training

### Root Mean Squared Error (RMSE)
```
RMSE = √MSE
```
- Same units as target (e.g., dollars)
- Easier to interpret: "predictions are off by ±$4,000 on average"

### Mean Absolute Error (MAE)
```
MAE = (1/n) × Σ|ŷᵢ - yᵢ|
```
- More robust to outliers than RMSE

### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_residual / SS_total)
```

| R² Value | Meaning |
|---|---|
| 1.0 | Perfect — model explains 100% of variance |
| 0.9+ | Excellent |
| 0.7–0.9 | Good |
| 0.5–0.7 | Moderate |
| < 0.5 | Poor |
| < 0 | Worse than predicting the mean |

---

## 📈 Output Charts

Running `python main.py` generates three charts in `./outputs/`:

### 1. `experiment1_salary.png`
- **Regression line** vs data points
- **Cost over epochs** — watch error fall as training progresses
- **Actual vs predicted** scatter — good models cluster near the diagonal

### 2. `experiment2_house_prices.png`
- **Actual vs predicted** prices
- **Residual plot** — errors should scatter randomly around zero
- **Cost curve** — convergence on 3-feature problem

### 3. `learning_rate_comparison.png`
- Side-by-side comparison of lr = 0.001, 0.01, 0.1, 0.5
- Visual proof of why learning rate matters

---

## 📋 Example Output

```
═══════════════════════════════════════════════════════════
  EXPERIMENT 1: Simple Linear Regression
  Predict Salary from Years of Experience
═══════════════════════════════════════════════════════════

  Train/Test Split: 160 train / 40 test  (80% / 20%)

  Epoch     1 / 500  |  MSE Cost: 5056505154.7992
  Epoch   101 / 500  |  MSE Cost:   53306135.3358
  Epoch   201 / 500  |  MSE Cost:   23538647.3169
  Epoch   500 / 500  |  MSE Cost:   14488550.7781

  ✅ Training complete! Final MSE: 14488550.7781

  EVALUATION METRICS — Test Set
  ──────────────────────────────────────────────────
  MSE  (Mean Squared Error)       :  16474102.7043
  RMSE (Root Mean Squared Error)  :     4058.8302
  MAE  (Mean Absolute Error)      :     3317.5897
  R²   (Coefficient of Det.)      :        0.9619
  ──────────────────────────────────────────────────
  Model Quality → Excellent (R² = 0.96)
```

---

## 🔬 What's Implemented

### Core Model (`src/linear_regression.py`)
- [x] Hypothesis function `ŷ = Xw + b`
- [x] MSE cost function
- [x] Manual gradient computation
- [x] Batch gradient descent training loop
- [x] Cost history tracking

### Preprocessing (`src/preprocessing.py`)
- [x] Synthetic dataset generation (salary & house prices)
- [x] Missing value handling (mean/median/zero imputation)
- [x] Min-Max normalisation (manual, no sklearn)
- [x] Standard scaling (Z-score)
- [x] Train/test split (manual shuffle)
- [x] Polynomial feature expansion

### Metrics (`src/metrics.py`)
- [x] MSE
- [x] RMSE
- [x] MAE
- [x] R² score
- [x] Formatted evaluation report

---

## 🚀 What to Try Next

| Challenge | What You'll Learn |
|---|---|
| Change `learning_rate` to 0.5 | Watch training become unstable |
| Set `n_epochs` to 50 | See what "early stopping" looks like |
| Add a 4th feature to house dataset | More features = richer model |
| Add polynomial features | Fit non-linear data with linear regression |
| Implement L2 regularisation | Prevent overfitting |
| Try mini-batch gradient descent | More efficient on large datasets |

---

## 📚 Math Reference

```
Hypothesis:         ŷ = X · w + b

MSE:                J(w,b) = (1/n) Σ(ŷᵢ - yᵢ)²

Gradient (weights): ∂J/∂w = (1/n) Xᵀ · (ŷ - y)

Gradient (bias):    ∂J/∂b = (1/n) Σ(ŷᵢ - yᵢ)

Weight update:      w ← w - α · ∂J/∂w

Bias update:        b ← b - α · ∂J/∂b

R²:                 R² = 1 - SS_res / SS_tot
```

---

## 🧰 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| NumPy | 1.21+ | Matrix operations |
| Pandas | 1.3+ | Data handling |
| Matplotlib | 3.4+ | Visualisations |

No ML frameworks. No shortcuts. Every formula is implemented by hand.

---

## 🌱 Learning Outcomes

After studying this project, you will be able to:

1. Explain how gradient descent optimises model parameters
2. Implement a regression model using only NumPy matrix operations
3. Interpret R², RMSE, and MAE metrics in context
4. Explain why feature normalisation matters for gradient descent
5. Identify underfitting and overfitting from train vs test metrics
6. Understand the bias-variance tradeoff conceptually

---

*Built for learning. Every line of code is a lesson.*
