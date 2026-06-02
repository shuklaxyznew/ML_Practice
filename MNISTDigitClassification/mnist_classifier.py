"""
MNIST Digit Classifier — Full Pipeline
=======================================
Steps: Load → Inspect → Normalize → Build → Train → Evaluate → Predict
Model: LeNet-style CNN (Conv2D × 2 → Dense → Softmax)
"""

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# Theory:  MNIST = 70k grayscale 28×28 images, 10 classes (0–9)
# Math:    X_train ∈ ℝ^(60000×28×28),  y_train ∈ {0,...,9}^60000
# ─────────────────────────────────────────────
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Train: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test : {X_test.shape},  Labels: {y_test.shape}")
print(f"Pixel range: {X_train.min()} – {X_train.max()}")


# ─────────────────────────────────────────────
# STEP 2 — UNDERSTAND IMAGES
# Theory:  Each image is a 2D matrix. Pixels ∈ [0,255].
# Math:    X[i] ∈ ℝ^(28×28),  class distribution p(k) ≈ 0.10 ∀k
# ─────────────────────────────────────────────
# Visualise 25 random samples
fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_train))
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(f"Label: {y_train[idx]}", fontsize=9)
    ax.axis('off')
plt.suptitle("Sample MNIST images", y=1.01)
plt.tight_layout()
plt.show()

# Pixel statistics
print(f"Mean pixel : {X_train.mean():.2f}")
print(f"Std  pixel : {X_train.std():.2f}")

# Class distribution
unique, counts = np.unique(y_train, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))


# ─────────────────────────────────────────────
# STEP 3 — NORMALIZE
# Theory:  Scale pixels to [0,1] → stable gradients, faster convergence
# Math:    X_norm = X / 255.0
#          Reshape: (N,28,28) → (N,28,28,1)  [channel dim for CNN]
# ─────────────────────────────────────────────
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# Add channel dimension required by Conv2D
X_train = X_train[..., tf.newaxis]   # (60000, 28, 28, 1)
X_test  = X_test[..., tf.newaxis]    # (10000, 28, 28, 1)

print(f"X_train: {X_train.shape}, dtype: {X_train.dtype}")
print(f"Pixel range after norm: {X_train.min():.2f} – {X_train.max():.2f}")


# ─────────────────────────────────────────────
# STEP 4 — BUILD MODEL (LeNet-style CNN)
# Theory:  Conv filters detect local patterns (edges, curves).
#          MaxPool downsamples. Dense classifies.
# Math:
#   Conv:    Z[i,j,f] = Σ W[m,n,c,f]·X[i+m,j+n,c] + b[f]
#   ReLU:    A = max(0, Z)
#   MaxPool: P[i,j] = max(2×2 window)
#   Softmax: ŷ_k = exp(z_k) / Σ exp(z_j)
#   Loss:    L = −log(ŷ_{y_true})
# ─────────────────────────────────────────────
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        # ── Block 1: detect edges and simple textures ──
        layers.Conv2D(32, (3,3), activation='relu', padding='same',
                      input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Block 2: detect curves, loops, strokes ──
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Classifier head ──
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')   # 10 digit classes
    ])
    return model

model = build_model()
model.summary()


# ─────────────────────────────────────────────
# STEP 5 — TRAIN
# Theory:  Mini-batch SGD with Adam optimizer.
#          Forward → loss → backward → weight update, every batch.
# Math (Adam):
#   m_t = β₁·m_{t-1} + (1−β₁)·g_t
#   v_t = β₂·v_{t-1} + (1−β₂)·g_t²
#   W_t = W_{t-1} − α · m̂_t / (√v̂_t + ε)
#   Default: α=0.001, β₁=0.9, β₂=0.999
# ─────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_mnist.keras', save_best_only=True,
        monitor='val_accuracy', verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'],     label='train acc')
ax1.plot(history.history['val_accuracy'], label='val acc')
ax1.set_title('Accuracy per epoch'); ax1.legend(); ax1.set_xlabel('Epoch')

ax2.plot(history.history['loss'],     label='train loss')
ax2.plot(history.history['val_loss'], label='val loss')
ax2.set_title('Loss per epoch'); ax2.legend(); ax2.set_xlabel('Epoch')
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────
# STEP 6 — EVALUATE
# Theory:  Accuracy, precision, recall, F1 per class.
#          Confusion matrix reveals which digits are confused.
# Math:
#   Accuracy  = Σ correct / N
#   Precision = TP / (TP + FP)
#   Recall    = TP / (TP + FN)
#   F1        = 2·P·R / (P+R)
# ─────────────────────────────────────────────
model.load_weights('best_mnist.keras')
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy : {test_acc:.4f}")
print(f"Test loss     : {test_loss:.4f}")

y_pred = model.predict(X_test).argmax(axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=[str(i) for i in range(10)]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix — MNIST CNN')
plt.show()

# Visualise worst mistakes
errors = np.where(y_pred != y_test)[0]
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
plt.suptitle("Misclassified samples")
for i, ax in enumerate(axes.flat):
    idx = errors[i]
    ax.imshow(X_test[idx,:,:,0], cmap='gray')
    ax.set_title(f"True:{y_test[idx]}  Pred:{y_pred[idx]}", fontsize=8)
    ax.axis('off')
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────
# STEP 7 — PREDICT
# Theory:  Model outputs softmax probabilities over 10 classes.
#          argmax gives the predicted digit; max gives confidence.
# Math:
#   ŷ = model(X) ∈ ℝ¹⁰      (probabilities, sum = 1)
#   class  = argmax_k(ŷ_k)
#   conf   = max(ŷ)
# ─────────────────────────────────────────────
def predict_and_show(index):
    img   = X_test[index]                      # (28,28,1)
    inp   = img[tf.newaxis, ...]               # (1,28,28,1)
    probs = model.predict(inp, verbose=0)[0]   # (10,)
    pred  = probs.argmax()
    conf  = probs.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(img[:,:,0], cmap='gray')
    ax1.set_title(f"True: {y_test[index]}  |  Pred: {pred}  ({conf:.1%})")
    ax1.axis('off')

    colors = ['green' if i == pred else 'steelblue' for i in range(10)]
    ax2.barh(range(10), probs, color=colors)
    ax2.set_yticks(range(10))
    ax2.set_xlabel('Probability')
    ax2.set_title('Class probabilities')
    plt.tight_layout(); plt.show()

# Try any index from 0–9999
predict_and_show(42)
predict_and_show(100)

# Batch predictions
all_probs = model.predict(X_test, batch_size=256)
all_preds = all_probs.argmax(axis=1)
print(f"\nFinal accuracy on 10,000 test images: {(all_preds == y_test).mean():.4f}")

# Save model
model.save('mnist_cnn_final.keras')
print("Model saved to mnist_cnn_final.keras")
