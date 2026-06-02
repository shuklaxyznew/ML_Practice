"""
metrics.py
==========
Manual implementations of regression evaluation metrics.

WHY DO WE NEED METRICS?
------------------------
After training, we need to know: "Is this model actually good?"
Metrics give us a standardised way to answer that question.

We implement:
  - MSE  : Mean Squared Error
  - RMSE : Root Mean Squared Error
  - MAE  : Mean Absolute Error
  - R²   : Coefficient of Determination
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error (MSE)

    Formula: MSE = (1/n) × Σ(ŷᵢ - yᵢ)²

    INTERPRETATION
    --------------
    - Units are squared (e.g., dollars² for salary prediction)
    - Lower is better; 0 is perfect
    - Penalises large errors more than small ones
    - Useful during training as a smooth loss function

    Parameters
    ----------
    y_true : array-like — actual values
    y_pred : array-like — predicted values

    Returns
    -------
    float : MSE score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_pred - y_true) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)

    Formula: RMSE = √MSE

    INTERPRETATION
    --------------
    - Same units as the target (e.g., dollars for salary prediction)
    - Much easier to interpret than MSE
    - A RMSE of $5,000 means "on average, predictions are off by $5,000"
    - Lower is better; 0 is perfect

    Parameters
    ----------
    y_true : array-like — actual values
    y_pred : array-like — predicted values

    Returns
    -------
    float : RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error (MAE)

    Formula: MAE = (1/n) × Σ|ŷᵢ - yᵢ|

    INTERPRETATION
    --------------
    - Same units as the target
    - Treats all errors equally (unlike MSE which penalises large ones more)
    - More robust to outliers than RMSE
    - Easy to explain: "average absolute prediction error"

    Parameters
    ----------
    y_true : array-like — actual values
    y_pred : array-like — predicted values

    Returns
    -------
    float : MAE score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_pred - y_true))


def r_squared(y_true, y_pred):
    """
    R-Squared (Coefficient of Determination)

    Formula:
        SS_res = Σ(yᵢ - ŷᵢ)²          ← residual sum of squares
        SS_tot = Σ(yᵢ - ȳ)²            ← total sum of squares
        R² = 1 - (SS_res / SS_tot)

    INTERPRETATION
    --------------
    R² answers: "How much of the variation in y does our model explain?"

    Examples:
        R² = 1.0  → Perfect! Model explains 100% of the variance
        R² = 0.9  → Great! Model explains 90% of the variance
        R² = 0.5  → Mediocre. Half the variance is unexplained
        R² = 0.0  → The model is as good as just predicting the mean
        R² < 0    → Worse than just predicting the mean — something's wrong!

    Think of it as a percentage: R² = 0.85 means your model explains
    85% of why the target values vary.

    Parameters
    ----------
    y_true : array-like — actual values
    y_pred : array-like — predicted values

    Returns
    -------
    float : R² score (typically between 0 and 1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_mean    = np.mean(y_true)
    ss_res    = np.sum((y_true - y_pred) ** 2)   # residual variance
    ss_tot    = np.sum((y_true - y_mean) ** 2)   # total variance

    # Edge case: if all y values are identical, avoid division by zero
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)


def print_metrics(y_true, y_pred, dataset_name=""):
    """
    Print a formatted evaluation report.

    Parameters
    ----------
    y_true       : array-like — actual values
    y_pred       : array-like — predicted values
    dataset_name : str        — label for the report (e.g., "Test Set")
    """
    mse  = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r_squared(y_true, y_pred)

    label = f" {dataset_name} " if dataset_name else ""
    print(f"\n{'─'*50}")
    print(f"  EVALUATION METRICS{label}")
    print(f"{'─'*50}")
    print(f"  MSE  (Mean Squared Error)       : {mse:>12.4f}")
    print(f"  RMSE (Root Mean Squared Error)  : {rmse:>12.4f}")
    print(f"  MAE  (Mean Absolute Error)      : {mae:>12.4f}")
    print(f"  R²   (Coefficient of Det.)      : {r2:>12.4f}")
    print(f"{'─'*50}")

    # Interpretation hints
    if r2 >= 0.9:
        quality = "Excellent"
    elif r2 >= 0.7:
        quality = "Good"
    elif r2 >= 0.5:
        quality = "Moderate"
    else:
        quality = "Poor — consider more features or a different model"

    print(f"\n  Model Quality → {quality} (R² = {r2:.2f})")
    print()

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
