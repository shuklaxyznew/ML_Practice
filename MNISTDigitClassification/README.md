# MNIST Handwritten Digit Classifier 🔢

A complete, beginner-friendly deep learning project that trains a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from the MNIST dataset. Built with TensorFlow/Keras, this project walks through every stage of the machine learning pipeline — from raw data loading to final model deployment — with detailed theory, mathematics, and code explanations at every step.

> **Expected accuracy:** ~99.2% on the 10,000-image test set after 5–10 epochs.

---

## Table of contents

- [Project overview](#project-overview)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Model architecture](#model-architecture)
- [Mathematics reference](#mathematics-reference)
- [Pipeline walkthrough](#pipeline-walkthrough)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the project](#running-the-project)
- [File outputs](#file-outputs)
- [Troubleshooting](#troubleshooting)
- [Concepts glossary](#concepts-glossary)
- [Acknowledgements](#acknowledgements)

---

## Project overview

This project demonstrates the full machine learning workflow applied to image classification:

```
Raw data → Exploration → Normalisation → Model design → Training → Evaluation → Inference
```

It is designed as a learning resource. Every code cell in the notebook is paired with:
- a **theory** explanation of *what* is happening and *why*
- a **mathematical** derivation of the underlying formula
- a **TensorFlow/Keras** implementation with inline comments

The model used is a **LeNet-style CNN** — two convolutional blocks followed by a fully connected classifier head. This architecture is simple enough to train on CPU in a few minutes yet powerful enough to reach near state-of-the-art accuracy on MNIST.

---

## Repository structure

```
mnist-digit-classifier/
│
├── README.md                   ← you are here
├── mnist_classifier.py         ← standalone Python script (all 7 steps)
├── mnist_classifier.ipynb      ← Jupyter notebook (recommended)
├── requirements.txt            ← pip dependencies
│
├── outputs/                    ← created automatically during training
│   ├── best_mnist.keras        ← best checkpoint saved by ModelCheckpoint
│   └── mnist_cnn_final.keras   ← final trained model
│
└── .gitignore                  ← excludes __pycache__, *.keras, etc.
```

---

## Dataset

**MNIST** (Modified National Institute of Standards and Technology) is the canonical beginner dataset for image classification.

| Property         | Value                          |
|------------------|-------------------------------|
| Total images     | 70,000                        |
| Training images  | 60,000                        |
| Test images      | 10,000                        |
| Image size       | 28 × 28 pixels                |
| Colour space     | Grayscale (1 channel)         |
| Pixel values     | Integer, range [0, 255]       |
| Classes          | 10 (digits 0 through 9)       |
| Class balance    | ~5,400–6,742 per class        |
| Source           | Yann LeCun, Corinna Cortes    |

The dataset is built into `tf.keras.datasets` and downloads automatically (~11 MB) on first use.

Each image represents one handwritten digit written by a different person. The spatial arrangement of bright pixels encodes the digit's shape. The model must learn which pixel patterns correspond to which digit.

---

## Model architecture

We use a **LeNet-style CNN** with two convolutional blocks and a fully connected classifier head.

```
Input (28 × 28 × 1)
        │
┌───────▼────────────────────────────────────┐
│  BLOCK 1 — edge & texture detection        │
│  Conv2D(32, 3×3, padding=same) → ReLU      │
│  BatchNormalization                         │
│  MaxPooling2D(2×2)  → output: 14×14×32    │
│  Dropout(0.25)                              │
└───────────────────────────────┬────────────┘
                                │
┌───────────────────────────────▼────────────┐
│  BLOCK 2 — curve & stroke detection        │
│  Conv2D(64, 3×3, padding=same) → ReLU      │
│  BatchNormalization                         │
│  MaxPooling2D(2×2)  → output:  7×7×64    │
│  Dropout(0.25)                              │
└───────────────────────────────┬────────────┘
                                │
                         Flatten → 3136
                                │
┌───────────────────────────────▼────────────┐
│  CLASSIFIER HEAD                           │
│  Dense(128) → ReLU                         │
│  BatchNormalization                         │
│  Dropout(0.5)                               │
│  Dense(10)  → Softmax                      │
└───────────────────────────────┬────────────┘
                                │
              Output: [p₀, p₁, ..., p₉]  (sums to 1.0)
```

### Layer-by-layer explanation

| Layer               | Purpose                                                                 | Output shape   |
|---------------------|-------------------------------------------------------------------------|----------------|
| `Conv2D(32, 3×3)`   | 32 learnable filters each scan a 3×3 window and detect low-level edges | (28, 28, 32)   |
| `BatchNormalization`| Normalises activations across the batch → stable, faster training       | (28, 28, 32)   |
| `MaxPooling2D(2×2)` | Takes max of each 2×2 region → halves spatial dimensions               | (14, 14, 32)   |
| `Dropout(0.25)`     | Randomly zeroes 25% of activations during training → fights overfitting | (14, 14, 32)   |
| `Conv2D(64, 3×3)`   | 64 deeper filters detect more complex patterns (curves, loops)          | (14, 14, 64)   |
| `BatchNormalization`| Stabilises deeper layer activations                                     | (14, 14, 64)   |
| `MaxPooling2D(2×2)` | Further spatial compression                                             | (7, 7, 64)     |
| `Dropout(0.25)`     | Regularisation                                                          | (7, 7, 64)     |
| `Flatten`           | Unrolls 3D tensor to 1D feature vector: 7×7×64 = 3,136 values          | (3136,)        |
| `Dense(128)`        | Fully connected: combines all features for high-level reasoning         | (128,)         |
| `BatchNormalization`| Normalises before final classification                                  | (128,)         |
| `Dropout(0.5)`      | Aggressive regularisation before the output layer                      | (128,)         |
| `Dense(10, softmax)`| Produces probability distribution over 10 digit classes                | (10,)          |

**Total trainable parameters: ~410,000** — lightweight enough to train on CPU.

---

## Mathematics reference

### Convolution operation

Each filter `f` at spatial position `(i, j)` computes a weighted sum over a local patch:

```
Z[i, j, f] = Σ_{m=0}^{2} Σ_{n=0}^{2} Σ_{c} W[m, n, c, f] · X[i+m, j+n, c]  +  b[f]
```

where `W` is the learnable filter weight tensor, `b` is the bias, and `c` is the input channel index.

### ReLU activation

Introduces non-linearity. Kills negative activations, preserves positives:

```
A = ReLU(Z) = max(0, Z)
```

### MaxPooling

Downsamples by retaining only the maximum value in each 2×2 window:

```
P[i, j] = max( Z[2i, 2j],  Z[2i, 2j+1],  Z[2i+1, 2j],  Z[2i+1, 2j+1] )
```

### Softmax (output layer)

Converts raw scores (logits) into a valid probability distribution:

```
ŷ_k = exp(z_k) / Σ_{j=0}^{9} exp(z_j)       for k = 0, 1, ..., 9

Properties:  ŷ_k ∈ (0, 1)   and   Σ_k ŷ_k = 1
```

### Sparse categorical cross-entropy loss

Measures how wrong the model's probability distribution is compared to the true label:

```
L = − (1/N) Σ_{i=1}^{N}  log( ŷ_{y_true,i} )
```

The model is penalised more heavily when it assigns low probability to the correct class.

### Backpropagation (chain rule)

Gradients are propagated backwards through every layer using the chain rule:

```
∂L/∂W = ∂L/∂ŷ  ·  ∂ŷ/∂Z  ·  ∂Z/∂W
```

Each layer multiplies its local gradient by the incoming gradient from the layer above.

### Adam optimiser

Adapts the learning rate per-parameter using running estimates of gradient moments:

```
g_t   = ∂L/∂W   (gradient at time t)

m_t   = β₁ · m_{t-1} + (1 − β₁) · g_t          ← 1st moment (momentum)
v_t   = β₂ · v_{t-1} + (1 − β₂) · g_t²          ← 2nd moment (variance)

m̂_t  = m_t / (1 − β₁ᵗ)                          ← bias-corrected
v̂_t  = v_t / (1 − β₂ᵗ)

W_t   = W_{t-1} − α · m̂_t / (√v̂_t + ε)

Defaults:  α = 0.001,  β₁ = 0.9,  β₂ = 0.999,  ε = 1×10⁻⁷
```

### Evaluation metrics

```
Accuracy  = (number of correct predictions) / N

Precision (class k) = TP_k / (TP_k + FP_k)
  → of all predictions labelled k, what fraction were truly k?

Recall (class k)    = TP_k / (TP_k + FN_k)
  → of all true k samples, what fraction did we correctly identify?

F1-score (class k)  = 2 · Precision_k · Recall_k / (Precision_k + Recall_k)
  → harmonic mean of precision and recall
```

---

## Pipeline walkthrough

### Step 1 — Load data

```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
```

Downloads and caches MNIST locally. Returns numpy arrays:
- `X_train`: shape `(60000, 28, 28)`, dtype `uint8`, values 0–255
- `y_train`: shape `(60000,)`, dtype `uint8`, values 0–9

### Step 2 — Understand images

Visualise random samples and check:
- Pixel value distribution (mean ≈ 33.3, std ≈ 78.6)
- Class balance (~5,400–6,742 per digit class)
- Label correctness (spot-check visually)

### Step 3 — Normalise

```python
X_train = X_train.astype('float32') / 255.0   # [0,255] → [0.0, 1.0]
X_train = X_train[..., tf.newaxis]             # add channel dim: (N,28,28,1)
```

Why: large raw values (0–255) produce large gradients that overshoot during weight updates. Normalising to [0, 1] keeps updates small and numerically stable, and ensures all input features start on equal footing.

### Step 4 — Build model

`build_model()` creates a `Sequential` CNN with two convolutional blocks + a dense classifier head. Key design choices:

- `padding='same'` keeps spatial dimensions constant after each conv layer
- `BatchNormalization` reduces internal covariate shift
- `Dropout` is a regulariser: randomly zeroes activations during training, forcing the network not to memorise

### Step 5 — Train

```python
model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=callbacks)
```

Three callbacks are used:
- `ModelCheckpoint` — saves weights whenever `val_accuracy` improves
- `ReduceLROnPlateau` — halves the learning rate if `val_loss` stops improving for 3 epochs
- `EarlyStopping` — stops training if `val_accuracy` stagnates for 5 epochs and restores the best weights

### Step 6 — Evaluate

Loads the best checkpoint and evaluates on the held-out test set (10,000 images the model has never seen). Outputs:
- Overall test accuracy and loss
- Per-class classification report (precision, recall, F1)
- Confusion matrix heatmap
- Gallery of misclassified samples

### Step 7 — Predict

```python
probs = model.predict(img[tf.newaxis, ...])[0]   # shape (10,)
pred  = probs.argmax()                            # predicted class
conf  = probs.max()                               # confidence
```

Produces a horizontal bar chart of all 10 class probabilities alongside the input image.

---

## Results

| Metric              | Value          |
|---------------------|----------------|
| Test accuracy       | ~99.2%         |
| Test loss           | ~0.03          |
| Typical error rate  | ~0.8% (≈80/10k)|
| Training time (CPU) | ~5–10 minutes  |
| Training time (GPU) | ~1–2 minutes   |

Common confusions (from the confusion matrix):
- **4 ↔ 9** — similar loop shape at the top
- **3 ↔ 8** — similar curve structure
- **1 ↔ 7** — similar vertical stroke

---

## Requirements

| Package       | Version   | Purpose                             |
|---------------|-----------|-------------------------------------|
| Python        | ≥ 3.9     | Runtime                             |
| TensorFlow    | ≥ 2.13    | Model building, training, inference |
| NumPy         | ≥ 1.24    | Array operations                    |
| Matplotlib    | ≥ 3.7     | Visualisation                       |
| Seaborn       | ≥ 0.12    | Confusion matrix heatmap            |
| scikit-learn  | ≥ 1.3     | Classification report, metrics      |
| Jupyter       | ≥ 1.0     | Running the notebook                |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mnist-digit-classifier.git
cd mnist-digit-classifier
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`** contents:
```
tensorflow>=2.13
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
jupyter>=1.0
```

> **GPU support:** If you have an NVIDIA GPU, replace `tensorflow` with `tensorflow[and-cuda]` for significantly faster training.

---

## Running the project

### Option A — Jupyter notebook (recommended)

```bash
jupyter notebook mnist_classifier.ipynb
```

Run cells one by one. Each cell contains theory, math, and code together — ideal for learning.

### Option B — Python script

```bash
python mnist_classifier.py
```

Runs all 7 steps sequentially. Plots will appear as pop-up windows.

### Option C — Google Colab

Upload `mnist_classifier.ipynb` to [colab.research.google.com](https://colab.research.google.com) and enable a GPU runtime:  
`Runtime → Change runtime type → T4 GPU`

---

## File outputs

After running the project, the following files will be created in the working directory:

| File                    | Description                                      |
|-------------------------|--------------------------------------------------|
| `best_mnist.keras`      | Best model weights (highest val_accuracy seen)   |
| `mnist_cnn_final.keras` | Final model after training completes             |

To reload a saved model:

```python
import tensorflow as tf
model = tf.keras.models.load_model('mnist_cnn_final.keras')
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**  
→ Run `pip install tensorflow` inside your activated virtual environment.

**Training is very slow**  
→ Use Google Colab with a GPU runtime, or reduce `epochs` to 5 for a quick test.

**`OOM` (out of memory) error during training**  
→ Reduce `batch_size` from 128 to 64 or 32.

**Plots not appearing in VS Code**  
→ Add `%matplotlib inline` at the top of the script, or use `plt.savefig('plot.png')` instead of `plt.show()`.

**`FileNotFoundError: best_mnist.keras`**  
→ The `ModelCheckpoint` callback creates this file during training. Run `model.fit(...)` before `model.load_weights(...)`.

---

## Concepts glossary

| Term                    | Definition                                                                                       |
|-------------------------|--------------------------------------------------------------------------------------------------|
| **Epoch**               | One full pass over the entire training dataset                                                   |
| **Batch**               | A subset of training samples processed together before one weight update                         |
| **Learning rate (α)**   | Step size for weight updates; too large → diverge, too small → slow convergence                  |
| **Overfitting**         | Model memorises training data but fails on new data; high train acc, low val acc                 |
| **Dropout**             | Regularisation technique: randomly zeros activations during training                            |
| **BatchNorm**           | Normalises layer inputs across the batch; reduces internal covariate shift                       |
| **Convolution**         | Sliding dot-product of a learnable filter across the input; detects local spatial patterns       |
| **MaxPooling**          | Reduces spatial dimensions by keeping only the maximum value in each local window               |
| **ReLU**                | Rectified Linear Unit: `max(0, x)` — the most common hidden-layer activation function           |
| **Softmax**             | Converts logits to probabilities; output sums to 1.0                                            |
| **Cross-entropy loss**  | Measures distance between predicted probability distribution and true label                      |
| **Backpropagation**     | Algorithm that computes gradients of the loss with respect to all model weights via chain rule   |
| **Adam**                | Adaptive gradient optimiser combining momentum and RMSProp; default for most deep learning tasks|
| **Confusion matrix**    | N×N grid showing predicted vs true class counts; diagonal = correct, off-diagonal = errors      |
| **Precision**           | TP / (TP + FP) — accuracy of positive predictions                                               |
| **Recall**              | TP / (TP + FN) — coverage of actual positives                                                   |
| **F1-score**            | Harmonic mean of precision and recall                                                            |

---

## Acknowledgements

- **Dataset:** Yann LeCun, Corinna Cortes, and Christopher Burges — [MNIST database](http://yann.lecun.com/exdb/mnist/)
- **Framework:** [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) by Google
- **Architecture inspiration:** LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998)

---

*Built as a deep learning learning project. Feel free to fork, extend, and experiment.*
