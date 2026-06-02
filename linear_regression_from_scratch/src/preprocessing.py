"""
preprocessing.py
================
Manual data preprocessing utilities — no scikit-learn transformers used.

WHAT IS PREPROCESSING?
-----------------------
Raw data is rarely in a format that machine learning models can use well.
Before training, we need to:

  1. Load and inspect the data
  2. Handle missing values
  3. Normalise features (bring them to the same scale)
  4. Split into training and test sets

WHY NORMALISATION MATTERS
--------------------------
Suppose your dataset has:
  - Feature A: house size in sq ft → values like 800, 1500, 3000
  - Feature B: number of bedrooms  → values like 1, 2, 4

Without normalisation, gradient descent will move much faster along
Feature A's axis (because changes there produce bigger gradients).
The result: training becomes slow and unstable.

After normalisation, all features are on a level playing field,
and gradient descent converges much faster.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATION (when no CSV is available)
# ─────────────────────────────────────────────────────────────────────────────
def generate_salary_dataset(n_samples=200, noise=5000, random_state=42):
    """
    Generate a synthetic salary-vs-experience dataset.

    Real-world intuition: "People with more years of experience tend to
    earn higher salaries — but there's natural variation."

    Parameters
    ----------
    n_samples    : int   — number of employees to generate
    noise        : float — salary randomness (standard deviation in $)
    random_state : int   — seed for reproducibility

    Returns
    -------
    df : DataFrame with columns ['YearsExperience', 'Salary']
    """
    rng = np.random.RandomState(random_state)

    years = rng.uniform(0.5, 15, n_samples)         # 0.5 to 15 years

    # True relationship: Salary = 30000 + 5000 × years + noise
    salary = 30_000 + 5_000 * years + rng.normal(0, noise, n_samples)

    df = pd.DataFrame({
        "YearsExperience": np.round(years, 1),
        "Salary": np.round(salary, 2)
    })

    print(f"  Generated salary dataset: {n_samples} samples")
    print(f"  True relationship: Salary ≈ 30,000 + 5,000 × YearsExperience + noise\n")
    return df


def generate_house_price_dataset(n_samples=300, random_state=42):
    """
    Generate a synthetic house price dataset with multiple features.

    Features:
        - Size     : house size in sq ft  (500–3500)
        - Bedrooms : number of bedrooms   (1–5)
        - Age      : age of house (years) (0–50)

    Target: Price in USD

    Parameters
    ----------
    n_samples    : int — number of houses to generate
    random_state : int — seed for reproducibility

    Returns
    -------
    df : DataFrame with columns ['Size', 'Bedrooms', 'Age', 'Price']
    """
    rng = np.random.RandomState(random_state)

    size     = rng.uniform(500, 3500, n_samples)
    bedrooms = rng.randint(1, 6, n_samples).astype(float)
    age      = rng.uniform(0, 50, n_samples)

    # True relationship (hidden from the model)
    price = (
        50_000
        + 120  * size
        + 8_000 * bedrooms
        - 500  * age
        + rng.normal(0, 20_000, n_samples)
    )

    df = pd.DataFrame({
        "Size":     np.round(size, 0),
        "Bedrooms": bedrooms,
        "Age":      np.round(age, 1),
        "Price":    np.round(price, 2)
    })

    print(f"  Generated house price dataset: {n_samples} samples")
    print(f"  Features: Size, Bedrooms, Age → Target: Price\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MISSING VALUE HANDLING
# ─────────────────────────────────────────────────────────────────────────────
def handle_missing_values(df, strategy="mean"):
    """
    Fill missing values in a DataFrame.

    WHY DOES THIS MATTER?
    ---------------------
    Most ML algorithms cannot handle NaN values. We need to either
    drop rows with missing data (losing information) or fill them
    with a reasonable substitute.

    Common strategies:
      - 'mean'   : Replace with column average (good for normally distributed)
      - 'median' : Replace with middle value (robust to outliers)
      - 'zero'   : Replace with 0 (careful — only use when it makes sense)

    Parameters
    ----------
    df       : DataFrame
    strategy : str — 'mean', 'median', or 'zero'

    Returns
    -------
    df : DataFrame (with NaNs filled)
    """
    missing_before = df.isnull().sum().sum()

    if missing_before == 0:
        print("  No missing values found.\n")
        return df

    print(f"  Found {missing_before} missing values. Strategy: '{strategy}'")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == "mean":
                fill_val = df[col].mean()
            elif strategy == "median":
                fill_val = df[col].median()
            else:  # zero
                fill_val = 0
            df[col] = df[col].fillna(fill_val)
            print(f"    Filled '{col}' with {strategy} = {fill_val:.2f}")

    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NORMALISATION — Min-Max Scaling
# ─────────────────────────────────────────────────────────────────────────────
class MinMaxScaler:
    """
    Scales each feature to the range [0, 1].

    Formula:
        X_scaled = (X - X_min) / (X_max - X_min)

    WHY [0, 1]?
    -----------
    When all features are in [0, 1], no single feature dominates the
    gradient. The model learns weights more evenly and gradient descent
    converges faster.

    IMPORTANT: Fit the scaler ONLY on training data.
    Then use the same min/max to transform both train and test data.
    This prevents data leakage — we pretend not to know the test set exists.
    """

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        """Learn min and max from training data."""
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        """Scale features using learned min/max."""
        if self.min_ is None:
            raise RuntimeError("Call .fit() before .transform()")
        range_ = self.max_ - self.min_
        # Avoid division by zero for constant features
        range_[range_ == 0] = 1
        return (X - self.min_) / range_

    def fit_transform(self, X):
        """Fit then transform in one step (use only on training data)."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Convert scaled values back to original scale."""
        return X_scaled * (self.max_ - self.min_) + self.min_


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NORMALISATION — Standard Scaler (Z-score)
# ─────────────────────────────────────────────────────────────────────────────
class StandardScaler:
    """
    Standardises features to have mean=0 and standard deviation=1.

    Formula:
        X_scaled = (X - μ) / σ

    WHEN TO USE THIS VS MIN-MAX?
    ----------------------------
    - StandardScaler: better when data has outliers or follows Gaussian dist.
    - MinMaxScaler  : better when you know features are bounded (e.g., age)

    Both work for linear regression — MinMaxScaler is more intuitive to teach.
    """

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Call .fit() before .transform()")
        std = self.std_.copy()
        std[std == 0] = 1
        return (X - self.mean_) / std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        return X_scaled * self.std_ + self.mean_


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split arrays into random train and test subsets.

    WHY DO WE SPLIT?
    ----------------
    If we trained and tested on the same data, we'd be cheating!
    The model could just memorise the training data and "score" perfectly
    without actually learning any useful patterns.

    By holding out a test set the model never sees during training,
    we get an honest measurement of how well the model generalises
    to new, unseen data.

    A typical split is 80% train / 20% test.

    Parameters
    ----------
    X           : ndarray (n_samples, n_features)
    y           : ndarray (n_samples,)
    test_size   : float — fraction for test set (e.g., 0.2 = 20%)
    random_state: int   — seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarrays
    """
    rng = np.random.RandomState(random_state)
    n   = len(y)

    # Shuffle indices
    indices   = np.arange(n)
    rng.shuffle(indices)

    # Determine split point
    n_test    = int(n * test_size)
    test_idx  = indices[:n_test]
    train_idx = indices[n_test:]

    print(f"  Train/Test Split: {len(train_idx)} train / {len(test_idx)} test")
    print(f"  ({100*(1-test_size):.0f}% / {100*test_size:.0f}%)\n")

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ─────────────────────────────────────────────────────────────────────────────
# POLYNOMIAL FEATURES (optional advanced feature)
# ─────────────────────────────────────────────────────────────────────────────
def add_polynomial_features(X, degree=2):
    """
    Add polynomial features to allow modelling of non-linear relationships.

    WHAT IS POLYNOMIAL REGRESSION?
    -------------------------------
    If your data follows a curve (not a straight line), a single feature X
    may not be enough. By adding X², X³, etc., we give linear regression
    the power to fit curves.

    Example for degree=2, 1 feature:
        [x] → [x, x²]

    The model is STILL linear in its parameters (weights are still linear),
    so we can train it the same way — we just added more engineered features.

    Parameters
    ----------
    X      : ndarray (n_samples, n_features)
    degree : int — highest polynomial degree to add

    Returns
    -------
    X_poly : ndarray (n_samples, n_features * degree)
    """
    X_poly = X.copy()
    for d in range(2, degree + 1):
        X_poly = np.hstack([X_poly, X ** d])
    print(f"  Added polynomial features up to degree {degree}")
    print(f"  Feature count: {X.shape[1]} → {X_poly.shape[1]}\n")
    return X_poly
