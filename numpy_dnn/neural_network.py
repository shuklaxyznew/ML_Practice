"""
Neural Network from Scratch — NumPy only
=========================================
Author  : Built with Claude (Anthropic)
License : MIT

Components implemented:
  1. Weight initialization (He / Xavier)
  2. Forward propagation
  3. Activation functions  (ReLU, Sigmoid, Tanh + derivatives)
  4. Loss functions        (Binary Cross-Entropy, MSE)
  5. Backpropagation       (chain rule, vectorised)
  6. Gradient descent      (vanilla SGD)
  7. Training loop
  8. Prediction & evaluation helpers
  9. Demo: XOR problem
"""

import numpy as np


# ─────────────────────────────────────────────
# Activation functions & their derivatives
# ─────────────────────────────────────────────

def relu(Z):
    """Rectified Linear Unit — fast, sparse, avoids vanishing gradient."""
    return np.maximum(0, Z)

def relu_derivative(Z):
    """Gradient of ReLU: 1 where Z > 0, else 0."""
    return (Z > 0).astype(float)

def sigmoid(Z):
    """Logistic sigmoid — squashes output to (0, 1). Used for binary output layer."""
    Z = np.clip(Z, -500, 500)          # numerical stability: prevents overflow in exp
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_derivative(Z):
    """Gradient of sigmoid: s * (1 - s)."""
    s = sigmoid(Z)
    return s * (1.0 - s)

def tanh_act(Z):
    """Hyperbolic tangent — zero-centred, stronger gradients than sigmoid."""
    return np.tanh(Z)

def tanh_derivative(Z):
    """Gradient of tanh: 1 - tanh²(Z)."""
    return 1.0 - np.tanh(Z) ** 2

def softmax(Z):
    """
    Softmax — converts raw scores to a probability distribution.
    Used for multi-class output layers.
    Subtracts max for numerical stability (log-sum-exp trick).
    """
    shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z   = np.exp(shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


# Map string names to (forward, derivative) pairs
ACTIVATIONS = {
    "relu":    (relu,     relu_derivative),
    "sigmoid": (sigmoid,  sigmoid_derivative),
    "tanh":    (tanh_act, tanh_derivative),
}


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

def binary_cross_entropy(Y_hat, Y):
    """
    Binary Cross-Entropy loss.

      L = -(1/m) * Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]

    Args:
        Y_hat : predictions,  shape (1, m)
        Y     : true labels,  shape (1, m)  — 0s and 1s
    Returns:
        scalar loss value
    """
    m     = Y.shape[1]
    Y_hat = np.clip(Y_hat, 1e-9, 1 - 1e-9)   # prevent log(0)
    loss  = -(1 / m) * np.sum(
        Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)
    )
    return float(loss)

def binary_cross_entropy_derivative(Y_hat, Y):
    """Gradient of BCE w.r.t. Y_hat."""
    Y_hat = np.clip(Y_hat, 1e-9, 1 - 1e-9)
    return -(Y / Y_hat) + (1 - Y) / (1 - Y_hat)

def mse(Y_hat, Y):
    """
    Mean Squared Error loss.

      L = (1/m) * Σ (ŷ - y)²

    Useful for regression tasks.
    """
    m = Y.shape[1]
    return float(np.sum((Y_hat - Y) ** 2) / m)

def mse_derivative(Y_hat, Y):
    """Gradient of MSE w.r.t. Y_hat."""
    m = Y.shape[1]
    return (2 / m) * (Y_hat - Y)


# ─────────────────────────────────────────────
# Neural Network class
# ─────────────────────────────────────────────

class NeuralNetwork:
    """
    Fully-connected feedforward neural network.

    Args:
        layer_dims  : list of integers — sizes of each layer including input.
                      e.g. [2, 4, 4, 1]  means:
                           input=2, hidden1=4, hidden2=4, output=1
        activations : list of activation names for each hidden layer.
                      Length = len(layer_dims) - 2.
                      Supported: 'relu', 'sigmoid', 'tanh'.
                      If None, all hidden layers default to 'relu'.
        output_act  : activation for the output layer.
                      'sigmoid' → binary classification
                      'softmax' → multi-class (not wired through ACTIVATIONS)
                      None      → linear output (regression)
        loss        : 'bce' (binary cross-entropy) or 'mse'

    Example:
        nn = NeuralNetwork(
            layer_dims  = [784, 128, 64, 10],
            activations = ['relu', 'relu'],
            output_act  = 'sigmoid',
            loss        = 'bce',
        )
    """

    def __init__(
        self,
        layer_dims,
        activations=None,
        output_act="sigmoid",
        loss="bce",
    ):
        self.layer_dims  = layer_dims
        self.L           = len(layer_dims) - 1   # number of weight layers
        self.output_act  = output_act
        self.loss_name   = loss

        # Default all hidden layers to ReLU
        if activations is None:
            activations = ["relu"] * (self.L - 1)

        # Store (forward_fn, backward_fn) per hidden layer
        self.act_fns = []
        for name in activations:
            if name not in ACTIVATIONS:
                raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}")
            self.act_fns.append(ACTIVATIONS[name])

        # Choose loss functions
        if loss == "bce":
            self.loss_fn  = binary_cross_entropy
            self.dloss_fn = binary_cross_entropy_derivative
        elif loss == "mse":
            self.loss_fn  = mse
            self.dloss_fn = mse_derivative
        else:
            raise ValueError(f"Unknown loss '{loss}'. Choose 'bce' or 'mse'.")

        # Initialize parameters
        self.params = {}
        self._init_params()

        # Will be populated during forward / backward passes
        self.cache = {}
        self.grads = {}

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_params(self):
        """
        He initialisation for ReLU layers:   W ~ N(0, sqrt(2/n_prev))
        Xavier initialisation for others:    W ~ N(0, sqrt(1/n_prev))

        He keeps variance stable through many ReLU layers.
        Biases start at 0.01 (not zero): a zero bias with zero-padded inputs causes
        pre-activations Z=0 exactly, landing on the non-differentiable kink of ReLU.
        This breaks gradient checks and can trigger dead neurons early in training.
        """
        for l in range(1, self.L + 1):
            n_prev = self.layer_dims[l - 1]
            n_curr = self.layer_dims[l]

            # He init: keeps activation variance stable across ReLU layers
            scale = np.sqrt(2.0 / n_prev)

            self.params[f"W{l}"] = np.random.randn(n_curr, n_prev) * scale
            self.params[f"b{l}"] = np.full((n_curr, 1), 0.01)

    # ── Forward propagation ─────────────────────────────────────────────────

    def forward(self, X):
        """
        Pass input X through every layer and return final predictions.

        Args:
            X : input array, shape (n_features, n_samples)
        Returns:
            A_L : output activations, shape (n_output, n_samples)
        """
        A = X
        self.cache["A0"] = X       # store input for backprop

        # Hidden layers
        for l in range(1, self.L):
            W, b   = self.params[f"W{l}"], self.params[f"b{l}"]
            Z      = W @ A + b                      # linear transform
            act_fn = self.act_fns[l - 1][0]         # forward activation
            A      = act_fn(Z)                      # non-linearity
            self.cache[f"Z{l}"] = Z
            self.cache[f"A{l}"] = A

        # Output layer
        W, b = self.params[f"W{self.L}"], self.params[f"b{self.L}"]
        Z    = W @ A + b
        self.cache[f"Z{self.L}"] = Z

        if self.output_act == "sigmoid":
            A = sigmoid(Z)
        elif self.output_act == "softmax":
            A = softmax(Z)
        else:
            A = Z                                   # linear / no activation

        self.cache[f"A{self.L}"] = A
        return A

    # ── Loss ────────────────────────────────────────────────────────────────

    def compute_loss(self, Y_hat, Y):
        """
        Compute scalar loss between predictions and true labels.

        Args:
            Y_hat : model predictions
            Y     : ground truth labels
        Returns:
            scalar float
        """
        return self.loss_fn(Y_hat, Y)

    # ── Backpropagation ─────────────────────────────────────────────────────

    def backward(self, X, Y):
        """
        Compute gradients of the loss w.r.t. all weights and biases
        via the chain rule (backpropagation).

        Populates self.grads with keys dW1, db1, ..., dWL, dbL.

        Args:
            X : input array,  shape (n_features, n_samples)
            Y : labels array, shape (n_output,   n_samples)
        """
        m    = X.shape[1]
        Y_hat = self.cache[f"A{self.L}"]

        # ── Output layer gradient ──────────────────────────────────────────
        # For sigmoid + BCE or softmax + cross-entropy, the combined
        # derivative simplifies elegantly to:  dZ_L = Y_hat - Y
        # This avoids numerical cancellation that would arise from
        # computing dA and then multiplying by the sigmoid derivative.
        if self.output_act in ("sigmoid", "softmax"):
            dZ = Y_hat - Y                          # shape: (n_output, m)
        else:
            # Linear output: chain dloss/dA * dA/dZ = dloss/dA * 1
            dA = self.dloss_fn(Y_hat, Y)
            dZ = dA                                 # identity derivative

        # ── Backpropagate through layers L → 1 ────────────────────────────
        for l in reversed(range(1, self.L + 1)):
            A_prev = self.cache[f"A{l-1}"]         # activations from layer below

            # Gradients of W and b for layer l
            # dW = (1/m) * dZ · A_prev^T
            # db = (1/m) * sum(dZ) over samples
            self.grads[f"dW{l}"] = (1 / m) * dZ @ A_prev.T
            self.grads[f"db{l}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # Propagate gradient to layer below (skip for l=1, no layer 0 weights)
            if l > 1:
                W        = self.params[f"W{l}"]
                dA_prev  = W.T @ dZ                     # chain rule: W^T · dZ
                act_deriv = self.act_fns[l - 2][1]      # derivative of hidden act
                dZ       = dA_prev * act_deriv(self.cache[f"Z{l-1}"])

    # ── Gradient descent ────────────────────────────────────────────────────

    def gradient_descent(self, learning_rate):
        """
        Vanilla (batch) gradient descent update rule:

            W = W - α · dW
            b = b - α · db

        Args:
            learning_rate : step size α (positive scalar)
        """
        for l in range(1, self.L + 1):
            self.params[f"W{l}"] -= learning_rate * self.grads[f"dW{l}"]
            self.params[f"b{l}"] -= learning_rate * self.grads[f"db{l}"]

    # ── Training loop ───────────────────────────────────────────────────────

    def train(self, X, Y, epochs=1000, learning_rate=0.01, print_every=100, verbose=True):
        """
        Full training loop:
            for each epoch:
                1. Forward pass  → predictions
                2. Compute loss
                3. Backward pass → gradients
                4. Update weights

        Args:
            X             : input,  shape (n_features, n_samples)
            Y             : labels, shape (n_output,   n_samples)
            epochs        : number of full passes over the dataset
            learning_rate : gradient descent step size
            print_every   : log loss every N epochs (0 = never)
            verbose       : whether to print training progress

        Returns:
            loss_history  : list of scalar losses, one per epoch
        """
        loss_history = []

        for epoch in range(1, epochs + 1):
            # 1. Forward
            Y_hat = self.forward(X)

            # 2. Loss
            loss = self.compute_loss(Y_hat, Y)
            loss_history.append(loss)

            # 3. Backward
            self.backward(X, Y)

            # 4. Update
            self.gradient_descent(learning_rate)

            # Logging
            if verbose and print_every > 0 and (epoch % print_every == 0 or epoch == 1):
                print(f"Epoch {epoch:>6d}/{epochs}  |  Loss: {loss:.6f}")

        return loss_history

    # ── Inference helpers ───────────────────────────────────────────────────

    def predict_proba(self, X):
        """
        Return raw output probabilities (or scores).

        Args:
            X : input array, shape (n_features, n_samples)
        Returns:
            array of shape (n_output, n_samples)
        """
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """
        Return class predictions.

        For binary output (sigmoid):  thresholds at `threshold`.
        For multi-class (softmax):    argmax over output neurons.

        Args:
            X         : input array, shape (n_features, n_samples)
            threshold : decision boundary for binary classification
        Returns:
            integer predictions, shape (1, n_samples) or (n_samples,)
        """
        proba = self.predict_proba(X)
        if self.output_act == "softmax":
            return np.argmax(proba, axis=0)
        return (proba >= threshold).astype(int)

    def accuracy(self, X, Y, threshold=0.5):
        """
        Compute classification accuracy.

        Args:
            X         : input array
            Y         : true labels
            threshold : for binary predictions
        Returns:
            accuracy as a float in [0, 1]
        """
        preds = self.predict(X, threshold)
        if self.output_act == "softmax":
            true = np.argmax(Y, axis=0)
        else:
            true = Y.flatten()
        return float(np.mean(preds.flatten() == true))

    # ── Gradient check (for debugging) ──────────────────────────────────────

    def gradient_check(self, X, Y, epsilon=1e-5, tolerance=1e-4):
        """
        Numerically verify that backprop gradients are correct.

        Uses the two-sided finite difference approximation per parameter:
            numerical_grad ≈ [L(θ + ε) - L(θ - ε)] / (2ε)

        Compares each numerical gradient to the analytical gradient from backprop.
        A max relative error < tolerance confirms backprop is correct.

        Strategy: compute all analytical gradients FIRST (one backward pass),
        then perturb each parameter individually for the numerical estimate.
        This avoids the issue of perturbed parameters contaminating cached grads.

        Note: gradient checks can show false failures when a pre-activation Z
        sits exactly at 0 (the non-differentiable kink of ReLU). This is why
        biases are initialised to 0.01 rather than 0.

        Args:
            X         : small input sample (fewer samples = faster check)
            Y         : corresponding labels
            epsilon   : perturbation magnitude (default 1e-5)
            tolerance : max acceptable relative error (default 1e-4)

        Returns:
            True if all gradients match within tolerance, False otherwise.
        """
        # Step 1: one forward + backward to get all analytical gradients
        self.forward(X)
        self.backward(X, Y)

        # Snapshot analytical grads before any perturbation touches the cache
        param_keys = [f"W{l}" for l in range(1, self.L + 1)] + \
                     [f"b{l}" for l in range(1, self.L + 1)]
        grad_keys  = [f"dW{l}" for l in range(1, self.L + 1)] + \
                     [f"db{l}" for l in range(1, self.L + 1)]
        stored_grads = {gk: self.grads[gk].copy()
                        for gk in grad_keys}

        # Step 2: iterate over every scalar parameter, perturb +/-eps, measure loss
        max_rel_error = 0.0

        for pk, gk in zip(param_keys, grad_keys):
            param    = self.params[pk]          # reference into self.params
            ana_grad = stored_grads[gk]

            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx      = it.multi_index
                orig_val = param[idx]

                # L(theta + eps)
                param[idx] = orig_val + epsilon
                loss_plus  = self.compute_loss(self.forward(X), Y)

                # L(theta - eps)
                param[idx] = orig_val - epsilon
                loss_minus = self.compute_loss(self.forward(X), Y)

                # Restore
                param[idx] = orig_val

                num  = (loss_plus - loss_minus) / (2 * epsilon)
                ana  = ana_grad[idx]
                rel  = abs(num - ana) / (abs(num) + abs(ana) + 1e-15)

                if rel > max_rel_error:
                    max_rel_error = rel

                it.iternext()

        status = "PASSED ✓" if max_rel_error < tolerance else "FAILED ✗"
        print(f"Gradient check {status}  |  max relative error: {max_rel_error:.2e}")
        return max_rel_error < tolerance


# ─────────────────────────────────────────────
# Demo — XOR problem
# ─────────────────────────────────────────────

def demo_xor():
    """
    XOR is the classic test for nonlinearity:
    a single-layer perceptron CANNOT solve it.
    A two-layer network with 4 hidden neurons solves it easily.

    Truth table:
        (0,0) → 0
        (0,1) → 1
        (1,0) → 1
        (1,1) → 0
    """
    print("=" * 50)
    print("Demo: XOR problem")
    print("=" * 50)

    np.random.seed(42)

    # Input: 4 examples, 2 features each — shape (2, 4)
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]], dtype=float)

    # Labels: XOR output — shape (1, 4)
    Y = np.array([[0, 1, 1, 0]], dtype=float)

    # Architecture: 2 → 4 → 1
    nn = NeuralNetwork(
        layer_dims  = [2, 4, 1],
        activations = ["relu"],
        output_act  = "sigmoid",
        loss        = "bce",
    )

    print(f"\nArchitecture  : {nn.layer_dims}")
    print(f"Hidden act    : relu")
    print(f"Output act    : sigmoid")
    print(f"Loss          : binary cross-entropy\n")

    # Gradient check on freshly initialised weights (BEFORE training).
    # ReLU is non-differentiable at Z=0; trained networks can drive some
    # neurons to Z~0, causing spurious finite-difference mismatches.
    print("Gradient check (fresh network):")
    nn.gradient_check(X, Y)
    print()

    # Train
    losses = nn.train(X, Y, epochs=5000, learning_rate=0.1, print_every=1000)

    # Results
    preds = nn.predict(X)
    proba = nn.predict_proba(X)
    acc   = nn.accuracy(X, Y)

    print("\nResults:")
    print(f"  Inputs      : {X.T.tolist()}")
    print(f"  True labels : {Y.flatten().tolist()}")
    print(f"  Predictions : {preds.flatten().tolist()}")
    print(f"  Probabilities: {np.round(proba.flatten(), 3).tolist()}")
    print(f"  Accuracy    : {acc * 100:.1f}%")
    print("\nFinal loss   :", round(losses[-1], 6))
    print("Initial loss :", round(losses[0],  6))
    print("=" * 50)
    return losses


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    demo_xor()
