# Neural Network from Scratch 🧠

A fully functional feedforward neural network built using **NumPy only** — no TensorFlow, no PyTorch, no Keras. Every component is implemented from first principles so you can see exactly what's happening under the hood.

---

## What's Inside

| Component | What it does |
|---|---|
| Weight initialisation | He init for ReLU, Xavier for sigmoid/tanh |
| Forward propagation | Vectorised layer-by-layer computation |
| Activation functions | ReLU, Sigmoid, Tanh, Softmax (+ derivatives) |
| Loss functions | Binary Cross-Entropy, Mean Squared Error |
| Backpropagation | Chain rule, fully vectorised |
| Gradient descent | Vanilla (batch) SGD |
| Gradient check | Numerical verification of backprop correctness |
| Demo | XOR problem solved end-to-end |

---

## Requirements

```
Python  >= 3.8
NumPy   >= 1.21
```

Install NumPy if you don't have it:

```bash
pip install numpy
```

No other dependencies.

---

## Quick Start

```bash
# Clone or download the file, then run the built-in demo
python neural_network.py
```

Expected output:

```
==================================================
Demo: XOR problem
==================================================

Architecture  : [2, 4, 1]
Hidden act    : relu
Output act    : sigmoid
Loss          : binary cross-entropy

Epoch      1/5000  |  Loss: 0.693147
Epoch   1000/5000  |  Loss: 0.184221
Epoch   2000/5000  |  Loss: 0.042183
Epoch   3000/5000  |  Loss: 0.018942
Epoch   4000/5000  |  Loss: 0.011203
Epoch   5000/5000  |  Loss: 0.006891

Results:
  Inputs       : [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
  True labels  : [0.0, 1.0, 1.0, 0.0]
  Predictions  : [0, 1, 1, 0]
  Probabilities: [0.007, 0.993, 0.993, 0.007]
  Accuracy     : 100.0%

Gradient check PASSED ✓  |  relative error: 3.21e-07
==================================================
```

---

## API Reference

### `NeuralNetwork(layer_dims, activations, output_act, loss)`

```python
from neural_network import NeuralNetwork

nn = NeuralNetwork(
    layer_dims  = [2, 4, 4, 1],   # input=2, hidden1=4, hidden2=4, output=1
    activations = ["relu", "relu"],# one activation per hidden layer
    output_act  = "sigmoid",       # output layer activation
    loss        = "bce",           # loss function
)
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `layer_dims` | `list[int]` | required | Sizes of each layer, including input |
| `activations` | `list[str]` | `["relu"] * n_hidden` | Hidden layer activations: `"relu"`, `"sigmoid"`, `"tanh"` |
| `output_act` | `str` | `"sigmoid"` | Output activation: `"sigmoid"`, `"softmax"`, or `None` (linear) |
| `loss` | `str` | `"bce"` | Loss function: `"bce"` (binary cross-entropy) or `"mse"` |

---

### `.train(X, Y, epochs, learning_rate, print_every, verbose)`

```python
losses = nn.train(
    X             = X_train,     # shape: (n_features, n_samples)
    Y             = Y_train,     # shape: (n_output,   n_samples)
    epochs        = 1000,
    learning_rate = 0.01,
    print_every   = 100,
    verbose       = True,
)
```

Returns a list of loss values, one per epoch.

> **Data format**: columns are samples, rows are features. This is the
> standard convention for vectorised neural networks and matches most
> linear algebra textbooks.

---

### `.predict(X, threshold)` and `.predict_proba(X)`

```python
proba = nn.predict_proba(X)        # raw probabilities,   shape (n_output, m)
preds = nn.predict(X, threshold=0.5)  # class predictions
acc   = nn.accuracy(X, Y)         # classification accuracy, float in [0,1]
```

---

### `.gradient_check(X, Y, epsilon, tolerance)`

Numerically verifies that backpropagation is computing the correct gradients using the two-sided finite difference approximation:

```
numerical_grad ≈ [L(θ + ε) - L(θ - ε)] / (2ε)
```

```python
passed = nn.gradient_check(X_small, Y_small)
# Gradient check PASSED ✓  |  relative error: 3.21e-07
```

Run this when implementing a new architecture or activation to confirm the math is right.

---

## Usage Examples

### Binary classification

```python
import numpy as np
from neural_network import NeuralNetwork

np.random.seed(0)

# Toy dataset: 2 features, 200 samples, binary labels
X = np.random.randn(2, 200)
Y = ((X[0] + X[1]) > 0).astype(float).reshape(1, -1)

nn = NeuralNetwork(
    layer_dims  = [2, 8, 4, 1],
    activations = ["relu", "relu"],
    output_act  = "sigmoid",
    loss        = "bce",
)

losses = nn.train(X, Y, epochs=2000, learning_rate=0.05)
print(f"Accuracy: {nn.accuracy(X, Y) * 100:.1f}%")
```

---

### Regression

```python
import numpy as np
from neural_network import NeuralNetwork

# Fit y = sin(x)
X = np.linspace(-np.pi, np.pi, 200).reshape(1, -1)
Y = np.sin(X)

nn = NeuralNetwork(
    layer_dims  = [1, 32, 16, 1],
    activations = ["tanh", "tanh"],
    output_act  = None,            # linear output for regression
    loss        = "mse",
)

losses = nn.train(X, Y, epochs=5000, learning_rate=0.001)
preds  = nn.predict_proba(X)      # raw scores (no thresholding)
```

---

### Multi-class (one-hot labels)

```python
import numpy as np
from neural_network import NeuralNetwork

# 3 classes, 4 features, 150 samples (e.g. Iris dataset)
n_samples  = 150
X = np.random.randn(4, n_samples)

# One-hot labels, shape (3, 150)
classes = np.random.randint(0, 3, n_samples)
Y = np.eye(3)[classes].T

nn = NeuralNetwork(
    layer_dims  = [4, 16, 8, 3],
    activations = ["relu", "relu"],
    output_act  = "softmax",
    loss        = "mse",           # use mse with softmax for simplicity
)

losses = nn.train(X, Y, epochs=1000, learning_rate=0.01)
preds  = nn.predict(X)            # returns argmax class indices
```

---

## How It Works

### The training loop

```
for each epoch:
    ┌─────────────────────────────────────────────┐
    │  1. Forward pass                            │
    │     X → [Z = W·A + b] → [A = act(Z)] → Ŷ  │
    │                                             │
    │  2. Compute loss                            │
    │     L = BCE(Ŷ, Y)                           │
    │                                             │
    │  3. Backward pass (backpropagation)         │
    │     dL/dW, dL/db  via chain rule            │
    │                                             │
    │  4. Gradient descent                        │
    │     W = W - α · dL/dW                       │
    │     b = b - α · dL/db                       │
    └─────────────────────────────────────────────┘
```

### Forward propagation

For each layer `l`:

```
Z[l] = W[l] · A[l-1] + b[l]
A[l] = activation(Z[l])
```

The intermediate values `Z[l]` and `A[l]` are cached because backpropagation needs them.

### Backpropagation

Starting from the output layer and working backwards using the chain rule:

```
dZ[L] = A[L] - Y                         (output layer, sigmoid + BCE)

for l = L down to 1:
    dW[l] = (1/m) · dZ[l] · A[l-1]ᵀ
    db[l] = (1/m) · sum(dZ[l])
    dA[l-1] = W[l]ᵀ · dZ[l]
    dZ[l-1] = dA[l-1] * activation'(Z[l-1])
```

### Weight initialisation

| Activation | Init | Formula |
|---|---|---|
| ReLU | He | `W ~ N(0, sqrt(2 / n_prev))` |
| Sigmoid / Tanh | Xavier | `W ~ N(0, sqrt(1 / n_prev))` |

He initialisation keeps the variance of activations stable across deep ReLU networks, preventing gradients from vanishing or exploding.

---

## Hyperparameter Guide

| Hyperparameter | Typical range | Effect |
|---|---|---|
| `learning_rate` | 0.001 – 0.1 | Too high → diverges. Too low → slow. Start at 0.01. |
| `epochs` | 500 – 10000 | More epochs = more training. Watch for overfitting. |
| Hidden neurons | 4 – 512 | More neurons = more capacity. Scale with dataset complexity. |
| Hidden layers | 1 – 5 | Deeper = more abstract features. Start shallow. |

---

## Project Structure

```
neural_network_from_scratch/
│
├── neural_network.py   ← Everything: network class, activations, losses, demo
└── README.md           ← This file
```

---

## Extending the Code

### Adding a new activation function

```python
# In neural_network.py, define the pair:
def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)

def leaky_relu_derivative(Z, alpha=0.01):
    return np.where(Z > 0, 1.0, alpha)

# Register it:
ACTIVATIONS["leaky_relu"] = (leaky_relu, leaky_relu_derivative)
```

### Adding momentum (SGD with momentum)

```python
# In gradient_descent(), maintain a velocity dict:
def gradient_descent(self, learning_rate, beta=0.9):
    for l in range(1, self.L + 1):
        self.velocity[f'W{l}'] = beta * self.velocity[f'W{l}'] \
                                 + (1 - beta) * self.grads[f'dW{l}']
        self.params[f'W{l}']  -= learning_rate * self.velocity[f'W{l}']
        # same for b
```

---

## License

MIT — free to use, modify, and learn from.
