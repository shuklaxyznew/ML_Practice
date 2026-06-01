"""
model/architecture.py
---------------------
TensorFlow model definition using the Functional API.

Why Functional API over Sequential?
  - Supports multiple inputs/outputs (easy to extend later)
  - Exposes intermediate layers for SHAP, feature attribution
  - Mirrors how Transformer models are built in HuggingFace
  - Named layers make debugging and visualization cleaner

Architecture: 3-layer deep regression network with:
  - Batch Normalization (training stability)
  - Dropout (regularization)
  - He initialization (optimal for ReLU activations)
  - Linear output (regression — no sigmoid/softmax)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def build_model(
    num_features:   int   = 5,
    hidden_units:   tuple = (128, 64, 32),
    dropout_rate:   float = 0.3,
    l2_reg:         float = 1e-4,
    learning_rate:  float = 1e-3,
) -> keras.Model:
    """
    Build and compile the student score regression model.

    Architecture decisions:
      - 3 hidden layers: captures non-linear feature interactions
        without overfitting on a ~2000 sample dataset
      - BatchNorm before activation: stabilizes training, reduces
        sensitivity to learning rate choice
      - Dropout after activation: prevents co-adaptation of neurons
      - L2 regularization: additional weight decay for generalization
      - He uniform init: mathematically optimal for ReLU (preserves
        variance across layers)
      - Linear output: regression requires unbounded output

    Parameters
    ----------
    num_features  : Number of input features (must match pipeline output).
    hidden_units  : Tuple of neuron counts per hidden layer.
    dropout_rate  : Dropout probability (applied after each hidden layer).
    l2_reg        : L2 regularization coefficient.
    learning_rate : Initial learning rate for Adam optimizer.

    Returns
    -------
    Compiled Keras model ready for training.
    """

    # Input layer — explicit shape enforces contract with pipeline
    inputs = keras.Input(shape=(num_features,), name="features")
    x = inputs

    # Hidden layers
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            kernel_initializer="he_uniform",
            kernel_regularizer=regularizers.l2(l2_reg),
            use_bias=False,  # BatchNorm has its own bias (beta)
            name=f"dense_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    # Output — single neuron, linear activation for regression
    outputs = layers.Dense(1, activation="linear", name="score_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="StudentScorePredictor")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        ),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    logger.info(f"Model built. Parameters: {model.count_params():,}")
    return model


def build_model_wide(num_features: int = 5) -> keras.Model:
    """
    Wider, shallower variant — useful for ablation studies.
    Compare against build_model() to understand depth vs width trade-offs.
    """
    inputs = keras.Input(shape=(num_features,), name="features")
    x = layers.Dense(256, activation="relu", kernel_initializer="he_uniform", name="wide_dense_1")(inputs)
    x = layers.BatchNormalization(name="wide_bn_1")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu", kernel_initializer="he_uniform", name="wide_dense_2")(x)
    x = layers.BatchNormalization(name="wide_bn_2")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="linear", name="score_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="StudentScorePredictor_Wide")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


if __name__ == "__main__":
    model = build_model(num_features=5)
    model.summary()

    import numpy as np
    dummy_input  = np.random.randn(4, 5).astype(np.float32)
    dummy_output = model(dummy_input, training=False)
    print(f"\nForward pass OK. Output shape: {dummy_output.shape}")
    print(f"Sample predictions: {dummy_output.numpy().flatten()}")
