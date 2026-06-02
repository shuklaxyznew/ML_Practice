"""
main.py
=======
Linear Regression From Scratch — Main Runner

This script orchestrates two complete experiments:

  EXPERIMENT 1 — Simple Linear Regression (1 feature)
      Predict salary from years of experience.
      Great for visualising the regression line.

  EXPERIMENT 2 — Multiple Linear Regression (3 features)
      Predict house prices from size, bedrooms, and age.
      Demonstrates how the model handles multiple inputs.

Run:
    python main.py

Requirements:
    pip install numpy pandas matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # use non-interactive backend (saves PNGs)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Our modules ──────────────────────────────────────────────────────────────
from src.linear_regression import LinearRegression
from src.metrics           import print_metrics, r_squared, root_mean_squared_error
from src.preprocessing     import (
    generate_salary_dataset,
    generate_house_price_dataset,
    handle_missing_values,
    MinMaxScaler,
    train_test_split,
)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLE
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
ACCENT     = "#58a6ff"
ACCENT2    = "#f78166"
ACCENT3    = "#3fb950"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"

plt.rcParams.update({
    "figure.facecolor" : DARK_BG,
    "axes.facecolor"   : DARK_BG,
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : TEXT_COLOR,
    "xtick.color"      : TEXT_COLOR,
    "ytick.color"      : TEXT_COLOR,
    "text.color"       : TEXT_COLOR,
    "grid.color"       : GRID_COLOR,
    "grid.alpha"       : 0.5,
    "font.family"      : "monospace",
    "axes.titlecolor"  : TEXT_COLOR,
    "figure.dpi"       : 120,
})

os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██████╗ EXPERIMENT 1: SIMPLE LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def experiment_1_salary():
    print("\n" + "═"*60)
    print("  EXPERIMENT 1: Simple Linear Regression")
    print("  Predict Salary from Years of Experience")
    print("═"*60)

    # ── 1. Generate / Load Data ──────────────────────────────────────────────
    df = generate_salary_dataset(n_samples=200, noise=4000, random_state=42)
    df = handle_missing_values(df)

    print(df.head())
    print(f"\n  Shape: {df.shape}")
    print(f"  Salary range: ${df['Salary'].min():,.0f} — ${df['Salary'].max():,.0f}\n")

    # ── 2. Prepare Data ──────────────────────────────────────────────────────
    X = df[["YearsExperience"]].values   # shape (200, 1)
    y = df["Salary"].values              # shape (200,)

    # Normalise features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ── 3. Train ─────────────────────────────────────────────────────────────
    model = LinearRegression(learning_rate=0.1, n_epochs=500, verbose=True)
    model.fit(X_train, y_train)
    model.summary()

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    print_metrics(y_train, y_train_pred, "Training Set")
    print_metrics(y_test,  y_test_pred,  "Test Set")

    # ── 5. Visualise ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle(
        "Experiment 1 — Salary Prediction (Simple Linear Regression)",
        fontsize=14, color=TEXT_COLOR, y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ─── Plot A: Regression Line ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(
        X_train[:, 0], y_train,
        color=ACCENT, alpha=0.6, s=40, label="Training Data", zorder=3
    )
    ax1.scatter(
        X_test[:, 0], y_test,
        color=ACCENT2, alpha=0.8, s=50, marker="^", label="Test Data", zorder=3
    )

    # Generate smooth regression line
    x_line  = np.linspace(0, 1, 200).reshape(-1, 1)
    y_line  = model.predict(x_line)
    ax1.plot(x_line[:, 0], y_line, color=ACCENT3, linewidth=2.5,
             label="Regression Line", zorder=4)

    ax1.set_title("Regression Line vs Data Points", fontsize=12, pad=12)
    ax1.set_xlabel("Years of Experience (normalised)")
    ax1.set_ylabel("Salary ($)")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=TEXT_COLOR)
    ax1.grid(True)

    # ─── Plot B: Cost over Epochs ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(model.cost_history, color=ACCENT, linewidth=1.5)
    ax2.set_title("Cost (MSE) vs Epochs\n— Watch It Fall As The Model Learns —",
                  fontsize=10)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.grid(True)

    # ─── Plot C: Actual vs Predicted ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(y_test, y_test_pred, color=ACCENT, alpha=0.7, s=35, zorder=3)

    # Perfect prediction line
    min_v, max_v = min(y_test), max(y_test)
    ax3.plot([min_v, max_v], [min_v, max_v],
             color=ACCENT3, linestyle="--", linewidth=2, label="Perfect Prediction")
    ax3.set_title("Actual vs Predicted\n(closer to diagonal = better)", fontsize=10)
    ax3.set_xlabel("Actual Salary ($)")
    ax3.set_ylabel("Predicted Salary ($)")
    ax3.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=TEXT_COLOR)
    ax3.grid(True)

    out = "outputs/experiment1_salary.png"
    plt.savefig(out, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"\n  💾 Saved: {out}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ██████╗ EXPERIMENT 2: MULTIPLE LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def experiment_2_house_prices():
    print("\n" + "═"*60)
    print("  EXPERIMENT 2: Multiple Linear Regression")
    print("  Predict House Prices (Size + Bedrooms + Age)")
    print("═"*60)

    # ── 1. Generate / Load Data ──────────────────────────────────────────────
    df = generate_house_price_dataset(n_samples=300, random_state=42)
    df = handle_missing_values(df)

    print(df.head())
    print(f"\n  Shape: {df.shape}")
    print(f"  Price range: ${df['Price'].min():,.0f} — ${df['Price'].max():,.0f}\n")

    # ── 2. Prepare Data ──────────────────────────────────────────────────────
    feature_cols = ["Size", "Bedrooms", "Age"]
    X = df[feature_cols].values
    y = df["Price"].values

    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ── 3. Train ─────────────────────────────────────────────────────────────
    model = LinearRegression(learning_rate=0.1, n_epochs=1000, verbose=True)
    model.fit(X_train, y_train)
    model.summary()

    # Print what each weight means
    print("  WEIGHT INTERPRETATION")
    print(f"  {'─'*40}")
    for feat, w in zip(feature_cols, model.weights):
        direction = "↑ increases" if w > 0 else "↓ decreases"
        print(f"  {feat:<12} weight = {w:>10.1f}  → higher {feat} {direction} price")
    print()

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    print_metrics(y_train, y_train_pred, "Training Set")
    print_metrics(y_test,  y_test_pred,  "Test Set")

    # ── 5. Visualise ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=DARK_BG)
    fig.suptitle(
        "Experiment 2 — House Price Prediction (Multiple Linear Regression)",
        fontsize=13, color=TEXT_COLOR
    )

    # ─── A: Actual vs Predicted ───────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(y_test, y_test_pred, color=ACCENT, alpha=0.6, s=30, zorder=3)
    mn, mx = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    ax.plot([mn, mx], [mn, mx], color=ACCENT3, linestyle="--", linewidth=2,
            label="Perfect")
    ax.set_title("Actual vs Predicted Price", fontsize=11)
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=TEXT_COLOR)
    ax.grid(True)

    # ─── B: Residuals ────────────────────────────────────────────────────────
    ax = axes[1]
    residuals = y_test - y_test_pred
    ax.scatter(y_test_pred, residuals, color=ACCENT2, alpha=0.6, s=30, zorder=3)
    ax.axhline(0, color=ACCENT3, linestyle="--", linewidth=1.5)
    ax.set_title("Residual Plot\n(should be randomly scattered around 0)",
                 fontsize=10)
    ax.set_xlabel("Predicted Price ($)")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.grid(True)

    # ─── C: Cost over Epochs ─────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(model.cost_history, color=ACCENT, linewidth=1.5)
    ax.set_title("Cost (MSE) vs Epochs", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True)

    plt.tight_layout()
    out = "outputs/experiment2_house_prices.png"
    plt.savefig(out, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"\n  💾 Saved: {out}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE COMPARISON (advanced demo)
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_rate_comparison():
    """Show how learning rate affects training — the mountain metaphor in action."""
    print("\n" + "═"*60)
    print("  BONUS: Learning Rate Comparison")
    print("═"*60)

    df      = generate_salary_dataset(n_samples=200, noise=4000, random_state=42)
    X       = df[["YearsExperience"]].values
    y       = df["Salary"].values
    scaler  = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(X_scaled, y, random_state=42)

    learning_rates = [0.001, 0.01, 0.1, 0.5]
    colors         = [ACCENT, ACCENT2, ACCENT3, "#d29922"]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    for lr, color in zip(learning_rates, colors):
        m = LinearRegression(learning_rate=lr, n_epochs=300, verbose=False)
        m.fit(X_train, y_train)
        label = f"lr = {lr}"
        if lr == 0.001:
            label += " ← too slow (underfits)"
        elif lr == 0.5:
            label += " ← unstable (overshoots)"
        elif lr == 0.1:
            label += " ← just right ✓"
        ax.plot(m.cost_history, color=color, linewidth=2, label=label)

    ax.set_title("Learning Rate Comparison\n— How Step Size Affects Training —",
                 fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Cost")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=TEXT_COLOR)
    ax.grid(True)

    out = "outputs/learning_rate_comparison.png"
    plt.savefig(out, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"\n  💾 Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  LINEAR REGRESSION FROM SCRATCH")
    print("  Building ML Fundamentals Without Shortcuts")
    print("█"*60)

    experiment_1_salary()
    experiment_2_house_prices()
    plot_learning_rate_comparison()

    print("\n" + "█"*60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("  Outputs saved to: ./outputs/")
    print("█"*60 + "\n")
