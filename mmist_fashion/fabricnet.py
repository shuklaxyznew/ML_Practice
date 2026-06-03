"""
FabricNet — Fashion MNIST Classifier
=====================================
Author  : Built with Claude (Anthropic)
Dataset : Fashion MNIST (10 clothing categories, 70 000 images)
License : MIT

Concepts taught in this file:
  1. Dense Neural Networks   — fully-connected layers
  2. Activation Functions    — ReLU (hidden), Softmax (output)
  3. Overfitting             — what it is, how to detect it
  4. Regularisation (L2)     — penalise large weights
  5. Dropout                 — randomly silence neurons during training
  6. Batch Normalisation      — stabilise layer inputs
  7. Learning rate scheduling — decay LR when plateau detected

Fashion MNIST classes:
  0 T-shirt/top   1 Trouser      2 Pullover   3 Dress      4 Coat
  5 Sandal        6 Shirt        7 Sneaker    8 Bag        9 Ankle boot
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
]

NUM_CLASSES  = 10
INPUT_DIM    = 784    # 28 × 28 flattened
IMG_SHAPE    = (28, 28)


# ─────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────

def load_data():
    """
    Load and preprocess Fashion MNIST.

    Preprocessing steps:
      a) Normalise pixels [0,255] → [0.0, 1.0]
         Keeps weight magnitudes small → faster, more stable training.

      b) Flatten 28×28 → 784
         Dense (fully-connected) layers expect 1-D vectors.
         Convolutional layers would keep the 2-D shape — but we are
         teaching dense networks here.

      c) One-hot encode labels 0–9 → [0,0,1,0,...,0]
         Required by categorical_crossentropy loss.
         e.g. class 2 → [0,0,1,0,0,0,0,0,0,0]

    Returns:
        x_train : (60000, 784)  float32
        y_train : (60000, 10)   float32  one-hot
        x_test  : (10000, 784)  float32
        y_test  : (10000, 10)   float32  one-hot
        y_test_int : (10000,)   int  raw labels (for confusion matrix)
    """
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()

    # Normalise
    x_train = x_tr.astype("float32") / 255.0
    x_test  = x_te.astype("float32") / 255.0

    # Flatten
    x_train = x_train.reshape(-1, INPUT_DIM)
    x_test  = x_test.reshape(-1, INPUT_DIM)

    # Store raw integer labels before one-hot (needed for confusion matrix)
    y_test_int = y_te.copy()

    # One-hot encode
    y_train = keras.utils.to_categorical(y_tr, NUM_CLASSES)
    y_test  = keras.utils.to_categorical(y_te, NUM_CLASSES)

    print(f"x_train : {x_train.shape}  range [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f"y_train : {y_train.shape}  (one-hot)")
    print(f"x_test  : {x_test.shape}")
    return x_train, y_train, x_test, y_test, y_test_int


# ─────────────────────────────────────────────────────────────
# 2. DENSE NEURAL NETWORK
# ─────────────────────────────────────────────────────────────

"""
Dense Neural Network — what it is
───────────────────────────────────
A Dense (fully-connected) layer connects every input neuron to every
output neuron. If layer A has 784 neurons and layer B has 512, there
are 784 × 512 = 401,408 weight connections between them.

Each neuron computes:
    output = activation( W · input + b )

Where:
    W      = weight matrix  (learned during training)
    b      = bias vector    (learned during training)
    input  = activations from the previous layer
    output = activations sent to the next layer

Why "dense"? Every unit is "densely" connected to every other.
Compare: a Convolutional layer only connects to a small local patch.
"""


def build_baseline_model():
    """
    Baseline model — NO regularisation, NO dropout.

    We build this first deliberately to SHOW overfitting.
    High train accuracy, much lower val accuracy = overfitting.

    Architecture:
        Input (784)
          → Dense(512) + ReLU
          → Dense(256) + ReLU
          → Dense(10)  + Softmax   ← 10 class probabilities

    Returns:
        Compiled Keras Model
    """
    model = keras.Sequential([
        keras.Input(shape=(INPUT_DIM,), name="input"),

        layers.Dense(512, activation="relu", name="hidden_1"),
        layers.Dense(256, activation="relu", name="hidden_2"),
        layers.Dense(NUM_CLASSES, activation="softmax", name="output"),
    ], name="baseline")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_regularised_model(
    l2_lambda  : float = 1e-4,
    dropout_1  : float = 0.4,
    dropout_2  : float = 0.3,
):
    """
    Regularised model — L2 + Dropout + BatchNorm.

    This model adds three defences against overfitting:

    ── L2 Regularisation ──────────────────────────────────────
    Adds a penalty term to the loss function:
        Loss_total = CrossEntropy(y, ŷ)  +  λ · Σ w²

    The λ · Σ w² term penalises large weights. During backprop,
    the gradient of this term pushes weights toward zero unless
    the data strongly justifies them being large. This prevents
    the network from memorising training quirks via extreme weights.

    l2_lambda (λ) controls the strength:
        Too large → weights shrink too much → underfitting
        Too small → barely any effect
        Good range: 1e-5 to 1e-3

    ── Batch Normalisation ─────────────────────────────────────
    Normalises activations within each mini-batch to have zero mean
    and unit variance, then applies learned scale (γ) and shift (β).

    Benefits:
      • Faster convergence — each layer sees stable input distributions
      • Mild regularisation — the per-batch statistics add noise
      • Reduces sensitivity to weight initialisation
      • Place AFTER Dense, BEFORE activation (best practice)

    ── Dropout ─────────────────────────────────────────────────
    During TRAINING: randomly sets each neuron's output to 0 with
    probability `rate`. If rate=0.4, 40% of neurons are silenced
    each forward pass.

    During INFERENCE: all neurons are active, but outputs are scaled
    by (1 - rate) to keep expected values consistent.

    Why does this help?
      The network cannot rely on any single neuron — it must learn
      redundant representations spread across many neurons.
      This forces generalisation rather than memorisation.

      Think of it like a team where random members are absent each
      day. The team learns to function without depending on any
      single person.

    Args:
        l2_lambda : L2 weight decay coefficient
        dropout_1 : dropout rate after first hidden layer
        dropout_2 : dropout rate after second hidden layer

    Returns:
        Compiled Keras Model
    """
    reg = regularizers.L2(l2_lambda)

    model = keras.Sequential([
        keras.Input(shape=(INPUT_DIM,), name="input"),

        # Hidden layer 1
        layers.Dense(512, kernel_regularizer=reg, name="hidden_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Activation("relu", name="relu_1"),
        layers.Dropout(dropout_1, name="dropout_1"),

        # Hidden layer 2
        layers.Dense(256, kernel_regularizer=reg, name="hidden_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Activation("relu", name="relu_2"),
        layers.Dropout(dropout_2, name="dropout_2"),

        # Output layer — Softmax converts raw scores → probabilities
        layers.Dense(NUM_CLASSES, activation="softmax", name="output"),
    ], name="regularised")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────
# 3. ACTIVATION FUNCTIONS — EXPLAINED
# ─────────────────────────────────────────────────────────────

"""
Activation Functions in this project
──────────────────────────────────────

ReLU (hidden layers):
    f(x) = max(0, x)

    • Fast to compute — just a threshold at zero
    • No vanishing gradient for x > 0 (gradient = 1)
    • Sparse activation — many neurons output exactly 0
    • Used in: every hidden layer

Softmax (output layer — multi-class):
    f(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)

    • Converts raw scores → probabilities that sum to 1.0
    • Amplifies the largest score (via exp), suppresses small ones
    • Perfect for "pick exactly one of N classes"
    • e.g. [2.1, 0.3, 5.8] → [0.02, 0.01, 0.97]

Why NOT sigmoid for multi-class?
    Sigmoid outputs are independent — they don't sum to 1.
    You could get [0.9, 0.8, 0.95] which is meaningless as a
    probability distribution over 10 classes.
    Softmax enforces mutual exclusivity.

Loss Function — Categorical Cross-Entropy:
    L = -Σ yᵢ · log(ŷᵢ)

    Since y is one-hot, only the true class contributes:
    L = -log(ŷ_correct_class)

    If model assigns 0.95 to correct class → loss = -log(0.95) ≈ 0.05
    If model assigns 0.05 to correct class → loss = -log(0.05) ≈ 3.0
"""


# ─────────────────────────────────────────────────────────────
# 4. OVERFITTING — DETECTION & INTUITION
# ─────────────────────────────────────────────────────────────

"""
Overfitting — what it is and why it happens
────────────────────────────────────────────
Overfitting happens when a model learns the training data SO well
that it memorises noise and irrelevant patterns, causing poor
performance on new data.

Signs of overfitting (in training curves):
    • train accuracy >> val accuracy  (large gap)
    • train loss keeps falling but val loss rises
    • val accuracy plateaus early, train accuracy keeps climbing

Analogy: A student who memorises past exam papers verbatim.
They score 100% on practice papers but fail the real exam
because questions are phrased differently.

Why dense networks overfit easily:
    A Dense(512) layer has 784 × 512 + 512 = 401,920 parameters
    for just one layer. With 60,000 training samples, the model
    has MORE parameters than constraints — it has plenty of capacity
    to memorise rather than generalise.

Overfitting remedies used in this project:
    1. L2 regularisation   — penalise large weights
    2. Dropout             — force redundant representations
    3. Batch Normalisation — stabilise + mild noise
    4. Early stopping      — halt when val loss stops improving
    5. Learning rate decay — reduce LR when plateau detected
"""


# ─────────────────────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────────────────────

def get_callbacks(model_name: str = "model"):
    """
    Training callbacks — automatic training management.

    EarlyStopping:
        Monitors val_loss. If it doesn't improve for `patience`
        epochs, training stops and the best weights are restored.
        Prevents wasted compute and overfitting past the sweet spot.

    ReduceLROnPlateau:
        If val_loss doesn't improve for `patience` epochs,
        multiply the learning rate by `factor`.
        e.g. 0.001 → 0.0003 → 0.0001 as training progresses.
        Helps fine-tune as the model nears convergence.

    ModelCheckpoint:
        Saves weights whenever val_accuracy improves.
        Ensures you always have the best model even if training
        is interrupted.

    Returns:
        list of Keras Callback objects
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"{model_name}_best.weights.h5",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]


def train(model, x_train, y_train, x_test, y_test,
          epochs=50, batch_size=256):
    """
    Train the model.

    Batch size intuition:
        batch_size=256 means: take 256 samples, compute gradients,
        update weights, repeat. One epoch = 60000/256 ≈ 234 updates.

        Large batch  → stable gradients, faster per epoch, may miss sharp minima
        Small batch  → noisy gradients, slower per epoch, often better generalisation
        256 is a solid default for Fashion MNIST.

    Args:
        model      : compiled Keras model
        x_train    : (60000, 784) float32
        y_train    : (60000, 10)  one-hot
        x_test     : validation data
        y_test     : validation labels
        epochs     : max epochs (EarlyStopping may stop earlier)
        batch_size : samples per gradient update

    Returns:
        history : Keras History object
    """
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=get_callbacks(model.name),
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(model, x_test, y_test, y_test_int):
    """
    Evaluate model and print per-class metrics.

    Precision: of everything the model labelled as class X,
               what fraction actually was X?
    Recall:    of all actual class X samples,
               what fraction did the model correctly find?
    F1:        harmonic mean of precision and recall.

    Args:
        model      : trained model
        x_test     : test inputs
        y_test     : one-hot test labels
        y_test_int : integer test labels (for classification report)
    """
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{'─'*45}")
    print(f"  {model.name}  |  test loss: {loss:.4f}  |  test acc: {acc*100:.2f}%")
    print(f"{'─'*45}")

    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred_int   = np.argmax(y_pred_proba, axis=1)

    print(classification_report(
        y_test_int, y_pred_int,
        target_names=CLASS_NAMES,
        digits=3,
    ))
    return loss, acc, y_pred_int


# ─────────────────────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_overfitting_comparison(history_base, history_reg,
                                save="overfitting_comparison.png"):
    """
    Side-by-side training curves: baseline vs regularised.

    This is the most important plot in the whole project.
    It visually shows what overfitting looks like and how
    regularisation cures it.

    What to look for:
        Baseline:     train acc >> val acc  → overfitting
        Regularised:  train ≈ val acc       → healthy generalisation
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Overfitting vs Regularisation\n"
                 "Left: baseline (no defence)   Right: regularised",
                 fontsize=13, y=1.01)

    titles = ["Baseline", "Regularised"]
    hists  = [history_base, history_reg]

    for col, (title, h) in enumerate(zip(titles, hists)):
        ep = range(1, len(h.history["loss"]) + 1)

        # Accuracy
        axes[0, col].plot(ep, h.history["accuracy"],     label="Train", lw=2)
        axes[0, col].plot(ep, h.history["val_accuracy"], label="Val",   lw=2, ls="--")
        axes[0, col].set_title(f"{title} — Accuracy")
        axes[0, col].set_xlabel("Epoch"); axes[0, col].set_ylabel("Accuracy")
        axes[0, col].legend(); axes[0, col].grid(alpha=0.3)
        axes[0, col].set_ylim(0.5, 1.0)

        gap = max(h.history["accuracy"]) - max(h.history["val_accuracy"])
        axes[0, col].text(0.98, 0.05, f"Gap: {gap*100:.1f}%",
                          transform=axes[0, col].transAxes,
                          ha="right", fontsize=10,
                          color="red" if gap > 0.05 else "green")

        # Loss
        axes[1, col].plot(ep, h.history["loss"],     label="Train", lw=2)
        axes[1, col].plot(ep, h.history["val_loss"], label="Val",   lw=2, ls="--")
        axes[1, col].set_title(f"{title} — Loss")
        axes[1, col].set_xlabel("Epoch"); axes[1, col].set_ylabel("Loss")
        axes[1, col].legend(); axes[1, col].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save}")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
                          save="confusion_matrix.png"):
    """
    Plot a 10×10 confusion matrix.

    Each cell [i][j] = number of times class i was predicted as j.
    The diagonal = correct predictions.
    Off-diagonal = mistakes.

    Common confusions to look for in Fashion MNIST:
        Shirt vs T-shirt/top (similar shape)
        Coat vs Pullover     (similar shape)
        Sneaker vs Ankle boot (both shoes)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.4, ax=ax,
    )
    ax.set_title(f"{title}\n(normalised by true class)", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save}")


def plot_sample_predictions(model, x_test, y_test_int,
                            n=20, save="sample_predictions.png"):
    """
    Show n test images with predicted vs true labels.
    Green title = correct.  Red title = wrong.
    """
    indices = np.random.choice(len(x_test), n, replace=False)
    images  = x_test[indices]
    true_l  = y_test_int[indices]
    pred_l  = np.argmax(model.predict(images, verbose=0), axis=1)

    cols = 5
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    fig.suptitle("Sample predictions  (green = correct, red = wrong)", fontsize=11)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap="gray")
        correct = pred_l[i] == true_l[i]
        color   = "green" if correct else "red"
        ax.set_title(f"P: {CLASS_NAMES[pred_l[i]]}\nT: {CLASS_NAMES[true_l[i]]}",
                     fontsize=7, color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save}")


def plot_activation_functions(save="activation_functions.png"):
    """
    Plot ReLU and Softmax side by side for reference.
    Helps visualise what these functions actually do to numbers.
    """
    x = np.linspace(-4, 4, 300)

    relu    = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh_v  = np.tanh(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Activation functions used in this project", fontsize=12)

    ax1.plot(x, relu,    label="ReLU  max(0,x)",    lw=2.5)
    ax1.plot(x, sigmoid, label="Sigmoid 1/(1+e⁻ˣ)", lw=2, ls="--", alpha=0.7)
    ax1.plot(x, tanh_v,  label="Tanh",               lw=2, ls=":",  alpha=0.7)
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.axvline(0, color="gray", lw=0.5)
    ax1.set_title("Hidden layer activations")
    ax1.set_xlabel("Input z");  ax1.set_ylabel("Output f(z)")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Softmax demo on 5 logits
    logits  = np.array([-1.0, 0.5, 2.0, 0.8, -0.3])
    exp_log = np.exp(logits)
    softmax = exp_log / exp_log.sum()
    ax2.bar(range(5), logits,  label="Raw logits", alpha=0.6)
    ax2.bar(range(5), softmax, label="After Softmax → probabilities",
            alpha=0.8, color="coral")
    ax2.set_xticks(range(5))
    ax2.set_xticklabels([f"Class {i}" for i in range(5)])
    ax2.set_title("Softmax: logits → probabilities\n"
                  f"Sum of softmax outputs = {softmax.sum():.3f}")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save}")


def plot_weight_distributions(baseline, regularised,
                              save="weight_distributions.png"):
    """
    Compare weight distributions of baseline vs regularised model.

    L2 regularisation should produce a tighter distribution
    around zero — fewer extreme weights, less memorisation.
    """
    def get_weights(model):
        all_w = []
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                all_w.append(weights[0].ravel())
        return np.concatenate(all_w)

    w_base = get_weights(baseline)
    w_reg  = get_weights(regularised)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle("Weight distributions — L2 regularisation forces weights toward zero",
                 fontsize=12)

    ax1.hist(w_base, bins=80, color="steelblue", alpha=0.8)
    ax1.set_title(f"Baseline\nstd={w_base.std():.4f}, range=[{w_base.min():.2f},{w_base.max():.2f}]")
    ax1.set_xlabel("Weight value"); ax1.set_ylabel("Count")
    ax1.grid(alpha=0.3)

    ax2.hist(w_reg,  bins=80, color="coral",     alpha=0.8)
    ax2.set_title(f"Regularised (L2)\nstd={w_reg.std():.4f}, range=[{w_reg.min():.2f},{w_reg.max():.2f}]")
    ax2.set_xlabel("Weight value")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save}")


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────

def main(epochs: int = 50, batch_size: int = 256):
    """
    Full pipeline:
        1.  Load data
        2.  Visualise activation functions (reference)
        3.  Build + train baseline model    (shows overfitting)
        4.  Build + train regularised model (shows the fix)
        5.  Compare training curves         (the key lesson)
        6.  Evaluate both models
        7.  Plot confusion matrix
        8.  Show sample predictions
        9.  Compare weight distributions

    Args:
        epochs     : max training epochs
        batch_size : samples per gradient update
    """
    print("=" * 55)
    print("  FabricNet — Fashion MNIST Classifier")
    print("=" * 55)

    # 1. Data
    x_train, y_train, x_test, y_test, y_test_int = load_data()

    # 2. Activation function reference plot
    plot_activation_functions()

    # 3. Baseline model (intentionally no regularisation → will overfit)
    print("\n[1/2] Training BASELINE model (no regularisation)...")
    baseline = build_baseline_model()
    baseline.summary()
    history_base = train(baseline, x_train, y_train, x_test, y_test,
                         epochs=epochs, batch_size=batch_size)

    # 4. Regularised model
    print("\n[2/2] Training REGULARISED model (L2 + Dropout + BatchNorm)...")
    regularised = build_regularised_model()
    regularised.summary()
    history_reg = train(regularised, x_train, y_train, x_test, y_test,
                        epochs=epochs, batch_size=batch_size)

    # 5. Overfitting comparison — THE MAIN LESSON
    plot_overfitting_comparison(history_base, history_reg)

    # 6. Evaluate
    print("\n── Evaluation ─────────────────────────────────────")
    _, acc_base, pred_base = evaluate(baseline,    x_test, y_test, y_test_int)
    _, acc_reg,  pred_reg  = evaluate(regularised, x_test, y_test, y_test_int)

    print(f"\nAccuracy improvement: {(acc_reg - acc_base)*100:+.2f}%")

    # 7. Confusion matrices
    plot_confusion_matrix(y_test_int, pred_base,
                          title="Baseline — Confusion Matrix",
                          save="confusion_baseline.png")
    plot_confusion_matrix(y_test_int, pred_reg,
                          title="Regularised — Confusion Matrix",
                          save="confusion_regularised.png")

    # 8. Sample predictions
    plot_sample_predictions(regularised, x_test, y_test_int)

    # 9. Weight distributions
    plot_weight_distributions(baseline, regularised)

    print("\nDone. All plots saved to current directory.")
    return baseline, regularised, history_base, history_reg


if __name__ == "__main__":
    main(epochs=50, batch_size=256)
