"""
Autoencoder from Scratch — TensorFlow / Keras
==============================================
Author  : Built with Claude (Anthropic)
Dataset : MNIST handwritten digits (built-in, no download needed)
License : MIT

Architecture taught here:
  Input (784)
    └─► Encoder  : 784 → 256 → 128 → 64 → [latent_dim]
    └─► Latent space (bottleneck): compressed representation
    └─► Decoder  : [latent_dim] → 64 → 128 → 256 → 784
  Output (784)

Loss: Mean Squared Error (pixel-wise reconstruction)

Concepts covered:
  1. Encoder       — compresses input into a small vector
  2. Latent space  — the compact learned representation
  3. Decoder       — reconstructs input from the small vector
  4. Recon. loss   — how far the reconstruction is from the original
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────

def load_mnist():
    """
    Load MNIST and prepare it for the autoencoder.

    Key decisions:
      • Flatten 28×28 images → 784-dim vectors (dense autoencoder)
      • Normalise pixel values from [0, 255] → [0.0, 1.0]
        This keeps activations stable and makes MSE loss interpretable
        (MSE of 0.01 means average pixel error of 0.1 on a 0–1 scale)
      • We discard labels — autoencoders are UNSUPERVISED,
        they only need X (input), not Y (label)

    Returns:
        x_train : (60000, 784)  float32
        x_test  : (10000, 784)  float32
    """
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    # Normalise to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # Flatten: (N, 28, 28) → (N, 784)
    x_train = x_train.reshape(-1, 784)
    x_test  = x_test.reshape(-1, 784)

    print(f"Train shape : {x_train.shape}")   # (60000, 784)
    print(f"Test  shape : {x_test.shape}")    # (10000, 784)
    return x_train, x_test


# ─────────────────────────────────────────────────────────────
# 2. ENCODER
# ─────────────────────────────────────────────────────────────

def build_encoder(input_dim: int, latent_dim: int) -> Model:
    """
    The Encoder compresses a high-dimensional input into a small
    latent vector. Think of it as a "summariser" — it must learn
    to keep the most important information and throw away noise.

    Architecture:
        Input (784)
          → Dense(256) + ReLU      ← first compression
          → Dense(128) + ReLU      ← second compression
          → Dense(64)  + ReLU      ← third compression
          → Dense(latent_dim)      ← NO activation — raw values,
                                     decoder will interpret them

    Why no activation on the last layer?
        The latent space should be unconstrained — values can be
        anything. Adding sigmoid would squash them to [0,1], which
        limits the expressiveness of the representation.

    Why ReLU for hidden layers?
        Same reason as any deep net — introduces non-linearity so
        the encoder can learn complex mappings (not just linear PCA).

    Args:
        input_dim  : size of the input vector (784 for MNIST)
        latent_dim : size of the compressed representation

    Returns:
        A Keras Model that maps (batch, 784) → (batch, latent_dim)
    """
    inputs = keras.Input(shape=(input_dim,), name="encoder_input")

    # Progressively compress the representation
    x = layers.Dense(256, activation="relu", name="enc_dense_1")(inputs)
    x = layers.Dense(128, activation="relu", name="enc_dense_2")(x)
    x = layers.Dense(64,  activation="relu", name="enc_dense_3")(x)

    # Bottleneck: the actual latent vector — no activation
    latent = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = Model(inputs, latent, name="encoder")
    return encoder


# ─────────────────────────────────────────────────────────────
# 3. DECODER
# ─────────────────────────────────────────────────────────────

def build_decoder(latent_dim: int, output_dim: int) -> Model:
    """
    The Decoder is a mirror of the encoder. It takes the small
    latent vector and reconstructs the original input as closely
    as possible. Think of it as a "regenerator."

    Architecture:
        Latent (latent_dim)
          → Dense(64)  + ReLU      ← first expansion
          → Dense(128) + ReLU      ← second expansion
          → Dense(256) + ReLU      ← third expansion
          → Dense(784) + Sigmoid   ← output layer, values ∈ [0, 1]

    Why Sigmoid on the output?
        Our pixel values were normalised to [0, 1]. Sigmoid guarantees
        the output is also in [0, 1], so the reconstruction makes sense
        as pixel values. Without it, the decoder could output -5 or 100
        which can't be interpreted as a pixel.

    Args:
        latent_dim : size of the latent vector (must match encoder)
        output_dim : size of the reconstruction (784 for MNIST)

    Returns:
        A Keras Model that maps (batch, latent_dim) → (batch, 784)
    """
    latent_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")

    # Mirror the encoder in reverse — progressively expand
    x = layers.Dense(64,  activation="relu", name="dec_dense_1")(latent_inputs)
    x = layers.Dense(128, activation="relu", name="dec_dense_2")(x)
    x = layers.Dense(256, activation="relu", name="dec_dense_3")(x)

    # Final reconstruction — sigmoid to keep outputs in [0, 1]
    outputs = layers.Dense(output_dim, activation="sigmoid", name="reconstruction")(x)

    decoder = Model(latent_inputs, outputs, name="decoder")
    return decoder


# ─────────────────────────────────────────────────────────────
# 4. AUTOENCODER (ENCODER + DECODER COMBINED)
# ─────────────────────────────────────────────────────────────

def build_autoencoder(input_dim: int, latent_dim: int):
    """
    Combine the encoder and decoder into a single end-to-end model.

    During training, data flows:
        x  →  encoder  →  z  →  decoder  →  x̂

    The autoencoder is trained to minimise reconstruction loss:
        Loss = MSE(x, x̂) = mean((x - x̂)²)

    Crucially: the TARGET is the INPUT itself (x). This is what makes
    autoencoders "self-supervised" — no human labels needed.

    Args:
        input_dim  : original data dimension (784)
        latent_dim : compression dimension (e.g. 32)

    Returns:
        autoencoder : full Model (input → reconstruction)
        encoder     : just the encoder half
        decoder     : just the decoder half
    """
    encoder = build_encoder(input_dim, latent_dim)
    decoder = build_decoder(latent_dim, input_dim)

    # Connect them: input → encoder → decoder → output
    ae_input  = keras.Input(shape=(input_dim,), name="ae_input")
    encoded   = encoder(ae_input)
    decoded   = decoder(encoded)

    autoencoder = Model(ae_input, decoded, name="autoencoder")
    return autoencoder, encoder, decoder


# ─────────────────────────────────────────────────────────────
# 5. RECONSTRUCTION LOSS
# ─────────────────────────────────────────────────────────────

"""
Reconstruction Loss — the full story
─────────────────────────────────────
The autoencoder has ONE job: make its output look like its input.
The loss function measures how badly it's failing at that job.

We use Mean Squared Error (MSE):

    MSE = (1/n) * Σ (xᵢ - x̂ᵢ)²

Where:
    xᵢ  = original pixel value (normalised, in [0,1])
    x̂ᵢ = reconstructed pixel value (also in [0,1])
    n   = total number of pixels (784 for MNIST)

Why MSE and not Binary Cross-Entropy?
    BCE treats each pixel as a probability (0 or 1). But real images
    have continuous grayscale values — a pixel can be 0.47, not just
    0 or 1. MSE is a better fit for continuous-valued reconstructions.

    BCE is used when: pixels are binary (pure black or pure white)
    MSE is used when: pixels are continuous (grayscale, RGB floats)

What does the loss actually teach the network?
    If the latent_dim is too small (e.g. 2), the encoder can't store
    enough info → high reconstruction loss → network learns better
    compressions. If latent_dim is large (e.g. 512), the encoder can
    almost memorise each input → low loss but the latent space becomes
    meaningless (no compression learned).

    The sweet spot: small enough to force learning, large enough to
    reconstruct well. For MNIST, 32 dimensions works excellently.
"""

def reconstruction_loss_demo(x_orig, x_recon):
    """
    Manually compute MSE reconstruction loss for one sample.
    This is exactly what Keras computes internally during training.

    Args:
        x_orig  : original input,     shape (784,)
        x_recon : reconstructed input, shape (784,)
    Returns:
        scalar MSE loss
    """
    diff       = x_orig - x_recon
    squared    = diff ** 2
    mse        = np.mean(squared)
    return float(mse)


# ─────────────────────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────────────────────

def train_autoencoder(
    autoencoder : Model,
    x_train     : np.ndarray,
    x_test      : np.ndarray,
    epochs      : int   = 30,
    batch_size  : int   = 256,
    learning_rate: float = 1e-3,
    save_path   : str   = "autoencoder_weights.weights.h5",
):
    """
    Compile and train the autoencoder.

    Key insight: target Y = input X
        autoencoder.fit(x_train, x_train)
    The model learns to reconstruct its own input.

    Optimiser: Adam
        Adaptive learning rate — adjusts per parameter.
        Works well out-of-the-box for autoencoders.

    Loss: MSE
        Penalises large pixel-level deviations.

    Callbacks:
        EarlyStopping  — stop if val_loss stops improving
                         (prevents overfitting and wasted compute)
        ModelCheckpoint — save the best weights automatically

    Args:
        autoencoder  : the full autoencoder model
        x_train      : training data, shape (60000, 784)
        x_test       : validation data, shape (10000, 784)
        epochs       : max training epochs
        batch_size   : samples per gradient update
        learning_rate: Adam step size
        save_path    : path to save best model weights

    Returns:
        history : Keras History object (contains loss curves)
    """
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",           # Mean Squared Error reconstruction loss
        metrics=["mae"],      # Mean Absolute Error — easier to interpret
    )

    autoencoder.summary()

    callbacks = [
        # Stop training if val_loss doesn't improve for 5 epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save weights whenever validation loss improves
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]

    # THE KEY LINE: target = input (self-supervised learning)
    history = autoencoder.fit(
        x_train, x_train,              # X = Y = same data
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nBest val_loss : {min(history.history['val_loss']):.6f}")
    return history


# ─────────────────────────────────────────────────────────────
# 7. VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────

def plot_loss_curves(history, save_path="loss_curves.png"):
    """
    Plot training and validation loss side by side.

    What to look for:
        • Both curves decreasing → model is learning
        • Val loss > train loss by a lot → overfitting
        • Both curves plateau → model has converged
        • Val loss spikes up → learning rate too high

    Args:
        history   : Keras History object from model.fit()
        save_path : where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.history["loss"]) + 1)

    ax1.plot(epochs, history.history["loss"],     label="Train MSE", linewidth=2)
    ax1.plot(epochs, history.history["val_loss"], label="Val MSE",   linewidth=2, linestyle="--")
    ax1.set_title("Reconstruction Loss (MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history.history["mae"],     label="Train MAE", linewidth=2)
    ax2.plot(epochs, history.history["val_mae"], label="Val MAE",   linewidth=2, linestyle="--")
    ax2.set_title("Mean Absolute Error")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Loss curves saved to {save_path}")


def plot_reconstructions(autoencoder, x_test, n=10, save_path="reconstructions.png"):
    """
    Show original images (top row) vs reconstructions (bottom row).
    Visually validates that the model learned to reconstruct.

    Args:
        autoencoder : trained autoencoder model
        x_test      : test images, shape (N, 784)
        n           : number of examples to display
        save_path   : where to save the figure
    """
    # Pick n random test images
    indices = np.random.choice(len(x_test), n, replace=False)
    samples = x_test[indices]

    # Get reconstructions
    reconstructed = autoencoder.predict(samples, verbose=0)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    fig.suptitle("Top: Original    Bottom: Reconstructed", fontsize=11)

    for i in range(n):
        # Original
        axes[0, i].imshow(samples[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=8)

        # Reconstruction
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Recon.", fontsize=8)

        # Show individual MSE below each pair
        mse = reconstruction_loss_demo(samples[i], reconstructed[i])
        axes[1, i].set_xlabel(f"MSE\n{mse:.4f}", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Reconstruction plot saved to {save_path}")


def plot_latent_space(encoder, x_test, y_test, save_path="latent_space.png"):
    """
    Visualise the 2D latent space coloured by digit class.

    Only works when latent_dim = 2. This is the most powerful
    visualisation — it shows whether the encoder has learned a
    structured, meaningful representation.

    What good looks like:
        • Each digit class forms a distinct cluster
        • Similar digits (e.g. 3 and 8) are near each other
        • Smooth transitions between clusters (no hard gaps)

    What bad looks like:
        • All points overlapping in one blob (under-trained)
        • Perfectly separated but random positions (no structure)

    Args:
        encoder   : trained encoder model
        x_test    : test images,  shape (10000, 784)
        y_test    : digit labels, shape (10000,)
        save_path : where to save the plot
    """
    # Encode all test images into latent space
    z = encoder.predict(x_test, verbose=0)

    if z.shape[1] != 2:
        print(f"Latent space has {z.shape[1]} dims — can only plot 2D latent space.")
        print("Re-run with latent_dim=2 to see this visualisation.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(z[:, 0], z[:, 1], c=y_test, cmap="tab10",
                         alpha=0.5, s=2)

    plt.colorbar(scatter, ax=ax, label="Digit class")
    ax.set_title("2D Latent Space — MNIST test set\n"
                 "(each colour = one digit class)")
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Latent space plot saved to {save_path}")


def plot_latent_interpolation(encoder, decoder, x_test, y_test,
                              digit_a=0, digit_b=1, steps=10,
                              save_path="interpolation.png"):
    """
    Walk through the latent space between two digit classes.
    This demonstrates the latent space is CONTINUOUS and smooth.

    How it works:
        1. Pick one example of digit_a, one of digit_b
        2. Encode both to get z_a and z_b
        3. Linearly interpolate: z = z_a + t*(z_b - z_a), t in [0,1]
        4. Decode each interpolated z back to an image
        5. Display the sequence — should show smooth morphing

    If the latent space is well-structured, the digits smoothly
    morph from digit_a into digit_b. This is IMPOSSIBLE with raw
    pixel interpolation, which just blurs between images.

    Args:
        encoder  : trained encoder
        decoder  : trained decoder
        x_test   : test images
        y_test   : digit labels
        digit_a  : starting digit class (0–9)
        digit_b  : ending digit class (0–9)
        steps    : number of interpolation steps
        save_path: where to save the plot
    """
    # Find one example of each digit
    idx_a = np.where(y_test == digit_a)[0][0]
    idx_b = np.where(y_test == digit_b)[0][0]

    # Encode to latent vectors
    z_a = encoder.predict(x_test[idx_a:idx_a+1], verbose=0)
    z_b = encoder.predict(x_test[idx_b:idx_b+1], verbose=0)

    # Linearly interpolate between the two latent vectors
    alphas = np.linspace(0, 1, steps)
    z_interp = np.array([z_a + alpha * (z_b - z_a) for alpha in alphas]).squeeze()

    # Decode each interpolated vector
    images = decoder.predict(z_interp, verbose=0)

    # Plot
    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.5, 2))
    fig.suptitle(f"Latent space interpolation: {digit_a} → {digit_b}", fontsize=11)

    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"α={alphas[i]:.1f}", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Interpolation plot saved to {save_path}")


# ─────────────────────────────────────────────────────────────
# 8. ANOMALY DETECTION (BONUS)
# ─────────────────────────────────────────────────────────────

def detect_anomalies(autoencoder, x_normal, x_anomaly,
                     threshold_percentile=95):
    """
    Use reconstruction loss as an anomaly score.

    Core idea:
        An autoencoder trained on normal data learns to reconstruct
        normal data well (low loss). When it sees an anomaly, it
        can't reconstruct it well → high reconstruction loss.
        High loss = likely anomaly.

    This is one of the most practical applications of autoencoders.
    Real uses: fraud detection, manufacturing defect detection,
               medical anomaly detection, network intrusion detection.

    Args:
        autoencoder          : trained autoencoder
        x_normal             : normal samples to establish threshold
        x_anomaly            : samples to test for anomalies
        threshold_percentile : percentile of normal losses to use as threshold

    Returns:
        threshold : scalar — loss above this = anomaly
        anomaly_scores : per-sample reconstruction loss for x_anomaly
        is_anomaly : boolean array, True where loss > threshold
    """
    # Compute per-sample reconstruction loss on normal data
    recon_normal   = autoencoder.predict(x_normal, verbose=0)
    losses_normal  = np.mean((x_normal - recon_normal) ** 2, axis=1)

    # Set threshold at the Nth percentile of normal losses
    threshold = np.percentile(losses_normal, threshold_percentile)
    print(f"Anomaly threshold ({threshold_percentile}th percentile): {threshold:.6f}")
    print(f"Normal data — mean loss: {losses_normal.mean():.6f}, "
          f"std: {losses_normal.std():.6f}")

    # Score the anomaly candidates
    recon_anomaly   = autoencoder.predict(x_anomaly, verbose=0)
    anomaly_scores  = np.mean((x_anomaly - recon_anomaly) ** 2, axis=1)
    is_anomaly      = anomaly_scores > threshold

    print(f"\nAnomaly data — mean loss: {anomaly_scores.mean():.6f}, "
          f"std: {anomaly_scores.std():.6f}")
    print(f"Detected as anomalies: {is_anomaly.sum()}/{len(is_anomaly)} "
          f"({100 * is_anomaly.mean():.1f}%)")

    return threshold, anomaly_scores, is_anomaly


# ─────────────────────────────────────────────────────────────
# 9. MAIN — RUN EVERYTHING
# ─────────────────────────────────────────────────────────────

def main(latent_dim: int = 32, epochs: int = 30):
    """
    Full pipeline:
        1. Load data
        2. Build model
        3. Train
        4. Visualise loss curves
        5. Visualise reconstructions
        6. Visualise latent space (if 2D)
        7. Demonstrate anomaly detection

    Args:
        latent_dim : compression size (try 2, 8, 32, 64)
        epochs     : maximum training epochs (EarlyStopping may cut short)
    """
    print("=" * 55)
    print(f"  Autoencoder — MNIST   |  latent_dim = {latent_dim}")
    print("=" * 55)

    # ── 1. Data ─────────────────────────────────────────────
    x_train, x_test = load_mnist()
    # Also load labels for latent space plot (not used in training)
    (_, y_train), (_, y_test) = keras.datasets.mnist.load_data()

    # ── 2. Build ─────────────────────────────────────────────
    autoencoder, encoder, decoder = build_autoencoder(
        input_dim=784,
        latent_dim=latent_dim,
    )

    # ── 3. Train ─────────────────────────────────────────────
    history = train_autoencoder(
        autoencoder=autoencoder,
        x_train=x_train,
        x_test=x_test,
        epochs=epochs,
        batch_size=256,
    )

    # ── 4. Loss curves ───────────────────────────────────────
    plot_loss_curves(history)

    # ── 5. Reconstructions ───────────────────────────────────
    plot_reconstructions(autoencoder, x_test, n=10)

    # ── 6. Latent space (only useful when latent_dim=2) ──────
    if latent_dim == 2:
        plot_latent_space(encoder, x_test, y_test)
        plot_latent_interpolation(encoder, decoder, x_test, y_test,
                                  digit_a=0, digit_b=1)

    # ── 7. Anomaly detection demo ────────────────────────────
    # Train: digits 0–4 are "normal"
    # Test:  digits 5–9 are "anomalies"
    normal_mask  = y_test <= 4
    anomaly_mask = y_test >= 5
    detect_anomalies(
        autoencoder,
        x_normal  = x_test[normal_mask][:500],
        x_anomaly = x_test[anomaly_mask][:500],
    )

    print("\nDone. All plots saved to current directory.")
    return autoencoder, encoder, decoder, history


if __name__ == "__main__":
    # Change latent_dim=2 to see the 2D latent space visualisation
    main(latent_dim=32, epochs=30)
