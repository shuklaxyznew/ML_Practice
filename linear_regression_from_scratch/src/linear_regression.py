"""
linear_regression.py
====================
Manual implementation of Linear Regression using only NumPy.

Author: Linear Regression From Scratch Project
Goal  : Understand machine learning fundamentals by building a model
        from zero — no scikit-learn, no TensorFlow, no shortcuts.

CORE IDEA
---------
Linear Regression finds the best straight line (or hyperplane) through
your data that minimizes prediction error. We do this by:

  1. Making predictions with our current weights (hypothesis function)
  2. Measuring how wrong those predictions are (cost function)
  3. Nudging the weights in the direction that reduces error (gradient descent)
  4. Repeating until the error barely changes (convergence)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(X, weights, bias):
    """
    The hypothesis function: ŷ = X·w + b

    This is the core of linear regression. We combine each feature (column in X)
    with a learned weight, then add a bias (intercept) term.

    Think of it like this:
        house_price = (size × w1) + (rooms × w2) + (location × w3) + b

    Parameters
    ----------
    X       : ndarray of shape (n_samples, n_features)
              Input feature matrix
    weights : ndarray of shape (n_features,)
              Learned coefficients — one per feature
    bias    : float
              The intercept / baseline prediction

    Returns
    -------
    predictions : ndarray of shape (n_samples,)
    """
    return X.dot(weights) + bias


# ─────────────────────────────────────────────────────────────────────────────
# COST FUNCTION — Mean Squared Error (MSE)
# ─────────────────────────────────────────────────────────────────────────────
def compute_cost(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE).

    Formula:
        MSE = (1 / n) × Σ(ŷᵢ - yᵢ)²

    WHY SQUARED?
    - Squaring makes all errors positive (no cancelling out)
    - It penalises large errors heavily (good!)
    - It's mathematically smooth — easy to differentiate

    A lower MSE means the model's predictions are closer to reality.

    Parameters
    ----------
    y_true : ndarray — actual target values
    y_pred : ndarray — model's predictions

    Returns
    -------
    mse : float
    """
    n = len(y_true)
    errors = y_pred - y_true
    mse = (1 / n) * np.sum(errors ** 2)
    return mse


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT DESCENT
# ─────────────────────────────────────────────────────────────────────────────
def compute_gradients(X, y_true, y_pred):
    """
    Compute the partial derivatives (gradients) of MSE with respect to
    weights and bias.

    WHAT IS A GRADIENT?
    -------------------
    A gradient tells us: "If I change this parameter slightly, how much
    does the error change — and in which direction?"

    We want to move in the OPPOSITE direction of the gradient (downhill),
    so we subtract it (scaled by learning rate) from our parameters.

    MATH DERIVATION
    ---------------
    For weights:   dL/dw = (2/n) × Xᵀ · (ŷ - y)
    For bias:      dL/db = (2/n) × Σ(ŷ - y)

    The factor of 2 cancels with the (1/2) often seen in textbooks.
    We simplify to (1/n) for cleanliness.

    Parameters
    ----------
    X      : ndarray (n_samples, n_features)
    y_true : ndarray (n_samples,)
    y_pred : ndarray (n_samples,)

    Returns
    -------
    dw : ndarray — gradient for weights
    db : float   — gradient for bias
    """
    n = len(y_true)
    error = y_pred - y_true            # how wrong are we? shape: (n,)
    dw = (1 / n) * X.T.dot(error)     # gradient for each weight
    db = (1 / n) * np.sum(error)      # gradient for the bias
    return dw, db


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS — Linear Regression
# ─────────────────────────────────────────────────────────────────────────────
class LinearRegression:
    """
    Linear Regression trained via Batch Gradient Descent.

    GRADIENT DESCENT — The Mountain Analogy
    ----------------------------------------
    Imagine a blindfolded person standing somewhere on a mountain, trying
    to reach the lowest point in the valley (minimum error).

    They can't see the full landscape, but they CAN feel the slope under
    their feet. So they take a small step downhill. Then another. And
    another. Eventually they reach the bottom — the point where the error
    is as small as possible.

    This is Gradient Descent. Our "position" is the model's weights,
    the "altitude" is the error, and each "step" is one update.

    Parameters
    ----------
    learning_rate : float (default 0.01)
        How big each step down the mountain is.
        Too large → overshoots the valley, bounces around.
        Too small → takes forever to reach the bottom.

    n_epochs : int (default 1000)
        How many times we loop through the full dataset.

    verbose : bool (default True)
        Print cost every 100 epochs so you can watch learning happen.
    """

    def __init__(self, learning_rate=0.01, n_epochs=1000, verbose=True):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.verbose = verbose

        # These will be learned during training
        self.weights = None
        self.bias = 0.0
        self.cost_history = []   # track error over time

    # ── TRAINING ─────────────────────────────────────────────────────────────
    def fit(self, X, y):
        """
        Train the model: find weights and bias that minimise MSE.

        THE TRAINING LOOP
        -----------------
        For each epoch (pass through data):
          1. Predict ŷ using current weights
          2. Compute cost (MSE) — how wrong are we?
          3. Compute gradients — which direction reduces error?
          4. Update weights and bias — take one step downhill
          5. Record cost — so we can visualise learning later

        Parameters
        ----------
        X : ndarray (n_samples, n_features) — training features
        y : ndarray (n_samples,)            — true target values
        """
        n_samples, n_features = X.shape

        # Initialise weights to zeros
        # (small random values also work — zeros is fine for linear regression)
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        print(f"\n{'─'*50}")
        print(f"  Training Linear Regression")
        print(f"  Learning Rate : {self.learning_rate}")
        print(f"  Epochs        : {self.n_epochs}")
        print(f"  Features      : {n_features}")
        print(f"  Samples       : {n_samples}")
        print(f"{'─'*50}\n")

        for epoch in range(self.n_epochs):
            # Step 1: Predict
            y_pred = predict(X, self.weights, self.bias)

            # Step 2: Compute cost
            cost = compute_cost(y, y_pred)
            self.cost_history.append(cost)

            # Step 3: Compute gradients
            dw, db = compute_gradients(X, y, y_pred)

            # Step 4: Update parameters (step downhill)
            #   new_weight = old_weight - learning_rate × gradient
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            # Step 5: Log progress
            if self.verbose and (epoch % 100 == 0 or epoch == self.n_epochs - 1):
                print(f"  Epoch {epoch+1:>5} / {self.n_epochs}  |  MSE Cost: {cost:.4f}")

        print(f"\n  ✅ Training complete! Final MSE: {self.cost_history[-1]:.4f}\n")

    # ── PREDICTION ───────────────────────────────────────────────────────────
    def predict(self, X):
        """
        Make predictions on new data using learned weights.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)

        Returns
        -------
        predictions : ndarray (n_samples,)
        """
        if self.weights is None:
            raise RuntimeError("Model not trained yet. Call .fit() first.")
        return predict(X, self.weights, self.bias)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    def summary(self):
        """Print a summary of the learned parameters."""
        print(f"\n{'─'*50}")
        print(f"  MODEL SUMMARY")
        print(f"{'─'*50}")
        print(f"  Bias (intercept): {self.bias:.4f}")
        for i, w in enumerate(self.weights):
            print(f"  Weight[{i}]       : {w:.4f}")
        print(f"{'─'*50}\n")
