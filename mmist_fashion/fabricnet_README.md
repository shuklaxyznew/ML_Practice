# FabricNet 👕

A complete Fashion MNIST classifier built to teach five core deep learning concepts from first principles: dense neural networks, activation functions, overfitting, regularisation, and dropout. Two models are trained deliberately — one to *show* overfitting, one to *fix* it.

---

## Project name

`FabricNet` — the network learns to recognise fabric and clothing categories. Short, domain-specific, and memorable.

---

## What you will learn

| Concept | What it is | Where in code |
|---|---|---|
| Dense neural network | Every neuron connects to every other | `build_baseline_model()` |
| Activation functions | ReLU (hidden), Softmax (output) | Every layer definition |
| Overfitting | Train acc >> val acc — memorisation | Baseline training curves |
| L2 regularisation | Penalise large weights via `λΣw²` | `kernel_regularizer=L2(1e-4)` |
| Dropout | Randomly silence neurons during training | `layers.Dropout(0.4)` |
| Batch normalisation | Stabilise mini-batch activations | `layers.BatchNormalization()` |
| Learning rate decay | Reduce LR when progress stalls | `ReduceLROnPlateau` callback |

---

## Dataset — Fashion MNIST

70,000 grayscale images, 28×28 pixels, 10 clothing categories:

```
0  T-shirt/top    1  Trouser    2  Pullover    3  Dress    4  Coat
5  Sandal         6  Shirt      7  Sneaker     8  Bag      9  Ankle boot
```

60,000 training images, 10,000 test images. Built into Keras — no manual download needed.

---

## Requirements

```
Python      >= 3.8
TensorFlow  >= 2.10
NumPy       >= 1.21
Matplotlib  >= 3.5
scikit-learn >= 1.0
seaborn     >= 0.12
```

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn jupyter
```

---

## Quick start

### Run the Python script
```bash
python fabricnet.py
```

Trains both models and saves eight plots to the current directory.

### Run the notebook
```bash
jupyter notebook fabricnet_notebook.ipynb
```

16 cells, top to bottom, with explanation before each one.

---

## Project structure

```
fabricnet/
├── fabricnet.py              ← Full implementation + main() runner
├── fabricnet_notebook.ipynb  ← 16-cell Jupyter notebook
└── README.md                 ← This file
```

---

## Architecture

### Baseline (no regularisation — intentionally overfits)

```
Input (784)
  → Dense(512) + ReLU
  → Dense(256) + ReLU
  → Dense(10)  + Softmax
```

### Regularised (L2 + Dropout + BatchNorm)

```
Input (784)
  → Dense(512) → BatchNorm → ReLU → Dropout(0.4)
  → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
  → Dense(10)  → Softmax
```

---

## Concepts explained

### Dense Neural Network

A Dense (fully-connected) layer connects every input neuron to every output neuron. For a layer with 784 inputs and 512 outputs:

```
Parameters = 784 × 512 (weights) + 512 (biases) = 401,920
```

Each neuron computes:
```
output = activation( W · input + b )
```

The weights `W` and biases `b` are what the network learns during training.

### Activation Functions

**ReLU — hidden layers:**
```
f(x) = max(0, x)
```
- Zeroes out negative values, passes positive values unchanged
- Gradient is either 0 (dead) or 1 (alive) — no vanishing gradient for positive inputs
- Fast to compute, causes sparse activation (many neurons output exactly 0)

**Softmax — output layer:**
```
f(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```
- Converts raw scores into probabilities that sum to exactly 1.0
- Amplifies the largest score, suppresses small ones
- Used when exactly one class is correct (mutually exclusive)

Why not sigmoid for multi-class? Sigmoid outputs are independent — you could get `[0.9, 0.8, 0.95]` which is meaningless as a probability distribution.

### Overfitting

Overfitting happens when a model learns the training data so well it memorises noise and irrelevant patterns, performing poorly on new data.

Signs in training curves:
- Train accuracy >> validation accuracy
- Train loss falls continuously while validation loss rises or plateaus
- Large growing gap between the two accuracy curves

Cause: The model has more parameters than the data can constrain. A `Dense(512)` layer has ~400K parameters. Given enough epochs and capacity, it will memorise rather than generalise.

### L2 Regularisation

Adds a penalty term to the loss:
```
Loss_total = CrossEntropy(y, ŷ) + λ · Σ w²
```

The `λ · Σ w²` term penalises large weights. Gradients from this term push weights toward zero unless the data strongly justifies keeping them large.

`λ` (lambda) controls strength:
- Too large → underfitting (weights over-shrunk)
- Too small → negligible effect
- Good range: `1e-5` to `1e-3`. Default: `1e-4`

### Dropout

During training: randomly set each neuron's output to 0 with probability `rate`. If `rate=0.4`, 40% of neurons are silenced each forward pass — different ones each time.

During inference: all neurons active, outputs scaled by `(1 - rate)` to maintain the same expected magnitude.

Why it helps: the network cannot rely on any single neuron. It must learn redundant representations spread across many neurons — forcing generalisation.

```python
layers.Dropout(0.4)   # after hidden layer 1
layers.Dropout(0.3)   # after hidden layer 2 (less aggressive, closer to output)
```

### Batch Normalisation

Normalises activations within each mini-batch to zero mean and unit variance, then applies learned scale (γ) and shift (β).

Benefits:
- Faster convergence — each layer sees stable input distributions
- Mild regularisation — per-batch statistics add training noise
- Less sensitive to weight initialisation

Place order: `Dense → BatchNorm → Activation → Dropout`

---

## Training callbacks

### EarlyStopping
```python
keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
```
Halts training when `val_loss` stops improving for 8 epochs. Prevents wasted compute and overfitting past the optimal point.

### ReduceLROnPlateau
```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4)
```
Reduces learning rate by 70% when `val_loss` stalls for 4 epochs.
Progression: `0.001 → 0.0003 → 0.0001` as training fine-tunes.

### ModelCheckpoint
```python
keras.callbacks.ModelCheckpoint(monitor='val_accuracy', save_best_only=True)
```
Saves weights whenever validation accuracy improves.

---

## Outputs

After running, eight files are saved:

| File | Contents |
|---|---|
| `activation_functions.png` | ReLU, Sigmoid, Softmax visualised |
| `overfitting_comparison.png` | THE key plot — baseline vs regularised curves |
| `confusion_baseline.png` | 10×10 confusion matrix, baseline model |
| `confusion_regularised.png` | 10×10 confusion matrix, regularised model |
| `sample_predictions.png` | 20 test images with predictions |
| `weight_distributions.png` | Baseline vs L2 weight histograms |
| `baseline_best.weights.h5` | Saved baseline weights |
| `regularised_best.weights.h5` | Saved regularised weights |

---

## Expected results

| Model | Test accuracy |
|---|---|
| Baseline (no regularisation) | ~88–89% |
| Regularised (L2 + Dropout + BN) | ~90–92% |

The gap in the training curves is more instructive than the final numbers. The baseline trains to ~97% accuracy but only validates at ~88% — a clear sign of memorisation. The regularised model stays within 1–2% across both sets.

---

## Experiments to try

### 1. Vary dropout rate
```python
build_regularised(drop1=0.5, drop2=0.4)   # more aggressive
build_regularised(drop1=0.2, drop2=0.1)   # lighter touch
```
Heavier dropout = more regularisation = trade accuracy for generalisation.

### 2. Vary L2 strength
```python
build_regularised(l2=1e-3)   # strong — may underfit
build_regularised(l2=1e-5)   # weak  — less effect
```

### 3. Add a third hidden layer
```python
layers.Dense(128, kernel_regularizer=reg)
layers.BatchNormalization()
layers.Activation('relu')
layers.Dropout(0.2)
```

### 4. Try different optimisers
```python
optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
optimizer=keras.optimizers.RMSprop(learning_rate=1e-3)
```

---

## What's next — CNNs

Dense networks treat each pixel independently. They have no sense of spatial structure — pixel (0,0) and pixel (14,14) are just two arbitrary input features with no relationship.

A Convolutional Neural Network (CNN) processes the 28×28 grid directly. Small filters slide across the image detecting edges, corners, and textures in local patches. These local features are then combined into global ones. CNNs typically reach 93–95% on Fashion MNIST vs 90–92% for dense networks, using fewer parameters.

---

## License

MIT — free to use, modify, and learn from.
