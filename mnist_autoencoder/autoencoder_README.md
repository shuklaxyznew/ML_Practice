# Autoencoder from Scratch 🔬

A complete, heavily-commented implementation of a **dense autoencoder** using TensorFlow/Keras, trained on MNIST handwritten digits. Every concept — encoder, latent space, decoder, reconstruction loss — is explained from first principles.

---

## What is an Autoencoder?

An autoencoder is a neural network that learns to compress data into a small representation and then reconstruct it back. It has one job: **make the output look like the input**.

```
Input (784 pixels)
  ──► Encoder  ──► Latent space (32 numbers)  ──► Decoder  ──► Output (784 pixels)
                        ↑
                  bottleneck: the compressed memory
```

The trick is the bottleneck. The network is *forced* to fit 784 pixels of information through only 32 numbers. To minimise reconstruction error, it must learn what is essential and what is noise. That is the learned representation.

---

## Concepts Covered

### 1. Encoder
The encoder is a compression function. It takes a high-dimensional input and maps it to a small latent vector.

```
Input (784)
  → Dense(256) + ReLU   ← first compression
  → Dense(128) + ReLU   ← second compression
  → Dense(64)  + ReLU   ← third compression
  → Dense(32)           ← latent vector (no activation)
```

Why no activation on the last encoder layer? The latent values should be unconstrained — the decoder will learn to interpret them. Applying sigmoid would squash everything to `[0,1]` and limit what the encoder can express.

### 2. Latent Space
The latent space is the compressed representation. Each image becomes a single point in 32-dimensional space.

Key properties of a well-trained latent space:
- Similar images map to nearby points
- Smooth transitions exist between clusters
- You can interpolate between two images by walking a straight line between their latent vectors

With `latent_dim=2`, the latent space can be visualised directly as a scatter plot. Each digit class forms a distinct cluster.

### 3. Decoder
The decoder is the mirror-image of the encoder — an expansion function.

```
Latent (32)
  → Dense(64)  + ReLU   ← first expansion
  → Dense(128) + ReLU   ← second expansion
  → Dense(256) + ReLU   ← third expansion
  → Dense(784) + Sigmoid ← output (sigmoid keeps values in [0,1])
```

Why sigmoid on the output? Pixel values were normalised to `[0,1]`. Sigmoid guarantees the output is in the same range, making the MSE loss meaningful.

### 4. Reconstruction Loss
The loss function measures how far the reconstruction is from the original:

```
MSE = (1/784) × Σ (original_pixel − reconstructed_pixel)²
```

The training call looks like:
```python
model.fit(x_train, x_train)   # input = target
```

This is **self-supervised learning** — no labels needed. The network teaches itself by trying to reproduce its own input.

Loss values to expect on MNIST:
- Random weights (untrained): MSE ≈ 0.17
- After training (latent_dim=32): MSE ≈ 0.01–0.02

---

## Requirements

```
Python      >= 3.8
TensorFlow  >= 2.10
NumPy       >= 1.21
Matplotlib  >= 3.5
scikit-learn >= 1.0   (for PCA in the notebook)
```

Install everything:
```bash
pip install tensorflow numpy matplotlib scikit-learn jupyter
```

---

## Quick Start

### Run the Python script
```bash
python autoencoder.py
```

This trains a full autoencoder with `latent_dim=32` and produces four plots:
- `loss_curves.png` — training and validation loss over epochs
- `reconstructions.png` — original vs reconstructed images
- `latent_space.png` — 2D PCA projection of the latent space
- `anomaly_detection.png` — reconstruction loss distribution

### Run the Jupyter notebook
```bash
jupyter notebook autoencoder_notebook.ipynb
```

The notebook has 15 cells covering every concept step by step, with outputs and explanations after each one.

---

## Project Structure

```
autoencoder/
├── autoencoder.py            ← Full implementation + main() runner
├── autoencoder_notebook.ipynb ← Step-by-step Jupyter notebook
└── README.md                 ← This file
```

---

## API Reference

### `build_encoder(input_dim, latent_dim) → Model`
```python
encoder = build_encoder(input_dim=784, latent_dim=32)
# Maps: (batch, 784) → (batch, 32)
```

### `build_decoder(latent_dim, output_dim) → Model`
```python
decoder = build_decoder(latent_dim=32, output_dim=784)
# Maps: (batch, 32) → (batch, 784)
```

### `build_autoencoder(input_dim, latent_dim) → (autoencoder, encoder, decoder)`
```python
autoencoder, encoder, decoder = build_autoencoder(
    input_dim=784,
    latent_dim=32,
)
```

### `train_autoencoder(...) → history`
```python
history = train_autoencoder(
    autoencoder=autoencoder,
    x_train=x_train,
    x_test=x_test,
    epochs=30,
    batch_size=256,
    learning_rate=1e-3,
)
```

### `detect_anomalies(autoencoder, x_normal, x_anomaly, threshold_percentile=95)`
```python
threshold, scores, is_anomaly = detect_anomalies(
    autoencoder,
    x_normal=x_normal_data,
    x_anomaly=x_test_data,
)
```

---

## Key Experiments to Try

### 1. Change latent dimension
```python
main(latent_dim=2)   # See the 2D latent space visualisation
main(latent_dim=8)   # Very compressed — blurry reconstructions
main(latent_dim=64)  # Less compressed — sharper reconstructions
```

Smaller `latent_dim` → more compression → blurrier output but more meaningful representation.  
Larger `latent_dim` → less compression → sharper output but weaker generalisation.

### 2. Visualise the 2D latent space
Set `latent_dim=2` in `main()`. The code automatically generates a 2D scatter plot with each digit coloured differently. You should see 10 distinct clusters forming.

### 3. Latent space interpolation
With a trained model, walk between two digit classes:
```python
plot_latent_interpolation(encoder, decoder, x_test, y_test,
                          digit_a=0, digit_b=1, steps=12)
```
The images should smoothly morph from one digit to another.

### 4. Anomaly detection
The autoencoder is trained on digits 0–4. Digits 5–9 are treated as anomalies. The reconstruction loss histogram shows clear separation between normal and anomalous samples.

---

## Architecture Diagram

```
Input (784)
    │
    ▼
┌───────────────────────────────────┐
│             ENCODER               │
│  Dense(256) → ReLU                │
│  Dense(128) → ReLU                │
│  Dense(64)  → ReLU                │
│  Dense(32)  → (no activation)     │
└───────────────┬───────────────────┘
                │
                ▼
         ┌────────────┐
         │   LATENT   │
         │   SPACE    │  ← 32 numbers encode the entire image
         │   (32-d)   │
         └──────┬─────┘
                │
                ▼
┌───────────────────────────────────┐
│             DECODER               │
│  Dense(64)  → ReLU                │
│  Dense(128) → ReLU                │
│  Dense(256) → ReLU                │
│  Dense(784) → Sigmoid             │
└───────────────────────────────────┘
                │
                ▼
           Output (784)
                │
                ▼
    ┌────────────────────────┐
    │   RECONSTRUCTION LOSS  │  Loss = MSE(Input, Output)
    │   MSE(x, x̂)           │  Gradient flows back through both
    └────────────────────────┘     encoder and decoder
```

---

## How Training Works — Step by Step

```
For each mini-batch of 256 images:

  Step 1 — Forward pass
    x (784)  →  encoder  →  z (32)  →  decoder  →  x̂ (784)

  Step 2 — Compute loss
    loss = MSE(x, x̂) = mean((x - x̂)²)

  Step 3 — Backward pass (backpropagation)
    Gradients flow from loss → decoder → z → encoder

  Step 4 — Update weights (Adam optimiser)
    Both encoder and decoder weights updated simultaneously
```

After many batches, the encoder learns to preserve the digits' essential structure, and the decoder learns to reconstruct from those compressed codes.

---

## From Autoencoder to VAE

A vanilla autoencoder has one limitation: the latent space has **gaps**. Random points sampled from it may decode to nonsense because the space between clusters is unexplored during training.

A Variational Autoencoder (VAE) solves this by:
- Encoding each input to a **distribution** `(μ, σ)` rather than a point `z`
- Sampling from that distribution during training
- Adding a KL-divergence loss term that regularises the latent space to follow a standard Gaussian

This forces the entire latent space to be covered, making it possible to **generate new data** by sampling `z ~ N(0, 1)` and decoding. That is the next step after this project.

---

## License

MIT — free to use, modify, and learn from.
