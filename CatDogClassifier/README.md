# 🐱🐶 Cat vs Dog Image Classifier — Complete CNN Guide

A complete **Computer Vision** project that builds a **Convolutional Neural Network (CNN)**  
from scratch to classify cat and dog photos. Every concept — convolution, filters, pooling,  
data augmentation — is explained with theory, mathematics, and heavily commented code.

> **Expected accuracy:** ~85% on 1,000 validation images (trained on just 2,000 images).  
> With the full 25,000-image dataset or transfer learning, expect 95%+.

---

## Table of contents

- [Why this is harder than MNIST](#why-this-is-harder-than-mnist)
- [What a CNN actually learns](#what-a-cnn-actually-learns)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Core concepts explained](#core-concepts-explained)
  - [Image preprocessing](#image-preprocessing)
  - [Convolution and filters](#convolution-and-filters)
  - [Pooling](#pooling)
  - [Data augmentation](#data-augmentation)
  - [CNN architecture](#cnn-architecture)
- [Mathematics reference](#mathematics-reference)
- [Pipeline walkthrough](#pipeline-walkthrough)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the project](#running-the-project)
- [File outputs](#file-outputs)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)
- [Glossary](#glossary)

---

## Why this is harder than MNIST

| Property | MNIST | Cat vs Dog |
|---|---|---|
| Image size | 28×28 px | 150×150 px |
| Colour | Grayscale (1 channel) | RGB (3 channels) |
| Subject position | Always centred | Anywhere in frame |
| Background | Always black | Complex, varied |
| Variation | Just different handwriting | Breeds, lighting, poses, occlusion |
| Total pixels per image | 784 | 67,500 |
| Expected accuracy (small dataset) | ~99% | ~85% |

---

## What a CNN actually learns

A CNN builds up understanding layer by layer — just like a human learning to recognise animals:

```
Layer 1  →  "I see a dark-to-light boundary here, a colour change there"
               (edges, colour gradients — simple local patterns)

Layer 2  →  "Those edges form a fuzzy texture, or a curved corner"
               (textures, fur patterns, simple shapes)

Layer 3  →  "Those textures and curves make a pointed ear shape, or a flat snout"
               (object parts — species-specific features)

Layer 4  →  "That arrangement of parts matches what I've seen labelled as 'cat'"
               (whole face / body structure)

Dense    →  "Output: 0.05" → "This is a cat (95% confident)"
```

---

## Repository structure

```
cat-vs-dog-classifier/
│
├── README.md                        ← you are here
├── cat_vs_dog_classifier.py         ← standalone Python script
├── cat_vs_dog_classifier.ipynb      ← Jupyter notebook (recommended)
├── requirements.txt                 ← pip dependencies
├── .gitignore                       ← excludes data/, model files, etc.
│
├── data/                            ← created automatically
│   ├── train/
│   │   ├── cats/                    ← 1,000 training cat images
│   │   └── dogs/                    ← 1,000 training dog images
│   └── validation/
│       ├── cats/                    ← 500 validation cat images
│       └── dogs/                    ← 500 validation dog images
│
└── outputs/                         ← created during training
    ├── best_cat_dog_model.keras     ← best checkpoint (by val_accuracy)
    └── cat_dog_cnn_final.keras      ← final model after training
```

---

## Dataset

We use the **Kaggle Dogs vs Cats** subset provided by Google's Machine Learning education team.

| Property | Value |
|---|---|
| Source | Google ML Education (Kaggle subset) |
| Total images used | 2,000 |
| Training images | 2,000 (1,000 cats + 1,000 dogs) |
| Validation images | 1,000 (500 cats + 500 dogs) |
| Image format | JPEG |
| Image size (raw) | Varies widely (640×480 to 3000×2000+) |
| Image size (after preprocessing) | 150×150 px |
| Colour | RGB (3 channels) |
| Download size | ~60 MB |
| Auto-download | Yes — handled in code |

The full Kaggle dataset has 25,000 images. We use a 2,000-image subset to keep training fast on CPU while demonstrating all the same concepts.

The dataset downloads automatically when you run the code — no manual steps required.

---

## Core concepts explained

### Image preprocessing

**The problem:** Images in the wild come in all shapes and sizes. A neural network requires  
fixed-size inputs because its weight matrices have fixed dimensions.

**Three preprocessing steps:**

1. **Resize** every image to 150×150 pixels  
   `target_size=(150, 150)` in `flow_from_directory()`

2. **Normalise** pixel values from [0, 255] to [0.0, 1.0]  
   `rescale=1./255` in `ImageDataGenerator`

3. **Channel dimension** — keep all 3 RGB channels  
   Shape becomes `(150, 150, 3)` per image, `(32, 150, 150, 3)` per batch

---

### Convolution and filters

A **convolution** is a sliding dot-product between a small matrix (the **filter**) and a patch  
of the image the same size.

The filter slides across every position in the image. At each position:
1. Multiply each filter value with the corresponding pixel value
2. Sum all the products
3. Write the result to the **feature map** (output)

**Example — vertical edge detection:**

```
Image patch P (3×3):        Filter W (3×3):
[ 10  20  30 ]              [-1  0  1]
[ 40  50  60 ]     ⊙        [-1  0  1]
[ 70  80  90 ]              [-1  0  1]

Z = (10×-1)+(20×0)+(30×1)+(40×-1)+(50×0)+(60×1)+(70×-1)+(80×0)+(90×1)
Z = -10+0+30-40+0+60-70+0+90 = 60   ← strong response = vertical edge!
```

**The key insight:** In a CNN, we do NOT design these filters by hand.  
We initialise them randomly and **gradient descent learns the optimal values automatically**.  
After training, filters in early layers look like edge detectors. Filters in deep layers  
detect complex shapes specific to cats and dogs.

**Multiple filters:**  
`Conv2D(32, 3×3)` means 32 different filters. Each learns to detect a different pattern.  
Output shape: `(H, W, 32)` — a stack of 32 feature maps.

---

### Pooling

`MaxPooling2D(2,2)` slides a 2×2 window across the feature map and keeps only the  
**maximum value** in each window, with stride 2 (non-overlapping windows).

```
Feature map (4×4):                  After MaxPool2D(2,2):
[ 1  3 | 2  4 ]                     [ 6  4 ]
[ 5  6 | 1  2 ]     →               [ 8  9 ]
───────────────
[ 7  1 | 9  3 ]
[ 2  8 | 4  6 ]
```

**Why max?** We care about *whether* a feature was detected, not *exactly where*  
in the 2×2 block. MaxPooling provides **translation invariance** — a cat's ear  
shifted 1 pixel is still detected as a cat's ear.

**Effect on shape:**  
`(148, 148, 32)` → after `MaxPool(2,2)` → `(74, 74, 32)`  
Spatial dimensions are halved. Channels unchanged.

---

### Data augmentation

With only 2,000 training images, the model risks **overfitting** — memorising the  
training images rather than learning general patterns.

Data augmentation applies random transformations to each image every time it is  
loaded during training. The model sees a slightly different version of every image  
on every epoch.

| Augmentation | What it does | Why it helps |
|---|---|---|
| `horizontal_flip=True` | Mirror left-right | Cats face either direction |
| `rotation_range=40` | Rotate 0–40° randomly | Tilted animals are still animals |
| `width_shift_range=0.2` | Shift left/right ≤20% | Subject not always centred |
| `height_shift_range=0.2` | Shift up/down ≤20% | Subject not always centred |
| `zoom_range=0.2` | Zoom in/out ≤20% | Animals at varying distances |
| `shear_range=0.2` | Shear transformation | Geometric variety |
| `fill_mode='nearest'` | Fill empty pixels | Realistic-looking edges |

**Critical rule:** Augmentation is applied **only to training data**.  
Validation data is never augmented — we need real, unmodified images to  
get an honest measure of performance.

---

### CNN architecture

```
Input image (150 × 150 × 3)    — 67,500 values
│
├─ Conv2D(32, 3×3) + ReLU      — 32 edge detectors
├─ MaxPool(2×2)                 — (74, 74, 32)
│
├─ Conv2D(64, 3×3) + ReLU      — 64 texture detectors
├─ MaxPool(2×2)                 — (36, 36, 64)
│
├─ Conv2D(128, 3×3) + ReLU     — 128 part detectors
├─ MaxPool(2×2)                 — (17, 17, 128)
│
├─ Conv2D(128, 3×3) + ReLU     — 128 structure detectors
├─ MaxPool(2×2)                 — (7, 7, 128)
│
├─ Flatten                      — 6,272 values
├─ Dropout(0.5)                 — regularisation
├─ Dense(512) + ReLU            — feature combination
│
└─ Dense(1) + Sigmoid           — P(dog) ∈ (0,1)
   < 0.5 → cat,   ≥ 0.5 → dog
```

**Why sigmoid + 1 neuron?**  
For binary classification (2 classes), a single output neuron with sigmoid is  
simpler and mathematically equivalent to using 2 neurons with softmax.

**Why double filters at each block (32 → 64 → 128)?**  
Deeper layers combine patterns from shallower layers, creating exponentially  
more complex representations. More filters = more representational capacity.

---

## Mathematics reference

### Convolution

For filter $f$ at position $(i, j)$ over input with $C$ channels:

$$Z[i, j, f] = \sum_{m=0}^{F-1} \sum_{n=0}^{F-1} \sum_{c=0}^{C-1} W[m, n, c, f] \cdot X[i+m,\; j+n,\; c] + b[f]$$

### Feature map size

$$\text{Output size} = \left\lfloor \frac{H - F + 2P}{S} \right\rfloor + 1$$

where $H$ = input size, $F$ = filter size, $P$ = padding, $S$ = stride.

### ReLU activation

$$A = \text{ReLU}(Z) = \max(0,\; Z)$$

### MaxPooling

$$P[i, j] = \max\!\left(Z[2i:2i+2,\; 2j:2j+2]\right)$$

### Sigmoid output

$$\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$$

### Binary cross-entropy loss

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### RMSprop update rule

$$v_t = \rho \cdot v_{t-1} + (1-\rho) \cdot g_t^2$$

$$W_t = W_{t-1} - \frac{\alpha}{\sqrt{v_t + \varepsilon}} \cdot g_t$$

Defaults: $\alpha = 10^{-4}$, $\rho = 0.9$, $\varepsilon = 10^{-7}$

### Dropout

During training, each activation is independently zeroed with probability $p$:

$$\tilde{h}_i = h_i \cdot \text{Bernoulli}(1-p)$$

During inference, all activations are active but scaled by $(1-p)$ for consistency.

---

## Pipeline walkthrough

### Step 1 — Dataset download and setup

One line downloads and extracts the dataset automatically:
```python
urllib.request.urlretrieve(url, 'cats_and_dogs_filtered.zip')
```
Images are copied into the `data/train/cats/`, `data/train/dogs/`, etc. structure  
that Keras's `flow_from_directory()` expects.

### Step 2 — Image preprocessing

```python
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    dir, target_size=(150,150), batch_size=32, class_mode='binary'
)
```
The generator reads images lazily from disk in batches — no need to load  
all 2,000 images into RAM at once.

### Step 3 — Data augmentation

Training generator applies random flips, rotations, zooms, and shifts to  
each image every time it is loaded. Validation generator is never augmented.

### Step 4 — Build model

```python
model = build_cat_dog_cnn()   # 4 Conv blocks + Dense head
model.summary()               # inspect layers and parameter counts
```

### Step 5 — Compile and train

```python
model.compile(optimizer=RMSprop(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=30, callbacks=callbacks)
```

Three callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`.

### Step 6 — Evaluate

Plot accuracy and loss curves. Load best checkpoint. Report final validation accuracy.  
Visualise learned filter weights and feature maps.

### Step 7 — Predict

```python
result = predict_image('my_cat.jpg', model)
# → {'label': 'CAT', 'confidence': 94.3, 'raw': 0.057}
```

---

## Results

| Metric | Value |
|---|---|
| Training images | 2,000 |
| Validation images | 1,000 |
| Expected validation accuracy | ~85% |
| Training time (CPU) | 15–30 minutes |
| Training time (GPU) | 3–5 minutes |
| Model size on disk | ~12 MB |

With improvements:
- Full 25,000-image dataset → ~92%
- Transfer learning (VGG16) → ~97%+

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.9 | Runtime |
| TensorFlow | ≥ 2.13 | Deep learning engine |
| NumPy | ≥ 1.24 | Array operations |
| Matplotlib | ≥ 3.7 | Visualisation |
| SciPy | ≥ 1.10 | Manual filter demo |
| Jupyter | ≥ 1.0 | Notebook environment |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cat-vs-dog-classifier.git
cd cat-vs-dog-classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**For GPU support (NVIDIA only):**
```bash
pip install tensorflow[and-cuda]
```

---

## Running the project

### Option A — Jupyter notebook (recommended for learning)

```bash
jupyter notebook cat_vs_dog_classifier.ipynb
```

Run cells one by one. Each cell has a theory markdown block above it  
explaining what the code does and why.

### Option B — Python script

```bash
python cat_vs_dog_classifier.py
```

Runs all steps sequentially. Plots appear as pop-up windows.

### Option C — Google Colab

Upload `cat_vs_dog_classifier.ipynb` to [colab.research.google.com](https://colab.research.google.com).  
Enable GPU: `Runtime → Change runtime type → T4 GPU`  
Training will be ~6× faster on GPU.

---

## File outputs

| File | Description |
|---|---|
| `data/` | Dataset images (auto-created) |
| `best_cat_dog_model.keras` | Best model weights by val_accuracy |
| `cat_dog_cnn_final.keras` | Final model after all training |

**To reload a saved model:**
```python
import tensorflow as tf
model = tf.keras.models.load_model('cat_dog_cnn_final.keras')
```

**To predict on your own image:**
```python
result = predict_image('path/to/your/image.jpg', model)
print(f"{result['label']} ({result['confidence']:.1f}% confident)")
```

---

## Troubleshooting

**Download fails or is slow**  
→ Download manually from https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip  
→ Place the zip file in the project root directory.

**`OOM` (out of memory) during training**  
→ Reduce `BATCH_SIZE` from 32 to 16 or 8.

**Training is very slow on CPU**  
→ Reduce `NUM_EPOCHS` to 10 for a quick test.  
→ Use Google Colab with GPU for proper training.

**`flow_from_directory` finds 0 images**  
→ Verify the folder structure: `data/train/cats/`, `data/train/dogs/`, etc.  
→ Run `download_and_prepare_dataset()` again.

**Accuracy stuck around 50%**  
→ This is random-chance performance for binary classification.  
→ Check normalisation — did you divide by 255.0?  
→ Check loss function — should be `binary_crossentropy` not `categorical_crossentropy`.

**`ModuleNotFoundError: scipy`**  
→ Run `pip install scipy` — needed for the manual filter visualisation cell.

---

## Next steps

Once you've completed this project, here are natural progressions:

**Transfer learning (biggest improvement for least effort)**  
Use VGG16 or ResNet50 pretrained on ImageNet. Fine-tune on cats/dogs.  
Expect 97%+ accuracy with the same 2,000 training images.
```python
base = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
```

**Grad-CAM (model explainability)**  
Visualise which pixels the model focused on when making its decision.  
A good model should highlight the face, not the background.

**Full Kaggle dataset**  
Download all 25,000 images from kaggle.com/c/dogs-vs-cats.  
Expect ~92% accuracy with the same CNN architecture.

**Multi-class classification**  
Extend to 37 breeds (Oxford Pets dataset) or 120 breeds (Stanford Dogs).  
Switch output to `Dense(N, softmax)` and loss to `categorical_crossentropy`.

**Model deployment**  
Wrap in a FastAPI endpoint and serve predictions over HTTP.  
Or build a Gradio UI: `pip install gradio` — 5 lines to a web demo.

---

## Glossary

| Term | Definition |
|---|---|
| **Convolution** | Sliding dot-product of a filter over an image; detects local spatial patterns |
| **Filter / Kernel** | Small weight matrix (e.g. 3×3) learned during training |
| **Feature map** | Output of applying one filter to the entire input; shows where that pattern appears |
| **ReLU** | `max(0, x)` — introduces non-linearity; prevents vanishing gradients |
| **MaxPooling** | Keeps max value in each 2×2 window; halves spatial dims; adds translation invariance |
| **Translation invariance** | Detecting a feature regardless of its exact position in the image |
| **Dropout** | Randomly zeros activations during training; prevents overfitting |
| **Overfitting** | Model memorises training data; high train acc, low val acc |
| **Data augmentation** | Random transformations to artificially increase training set diversity |
| **Sigmoid** | `1/(1+e^{-x})` — maps any number to (0,1); used for binary output |
| **Binary cross-entropy** | Loss function for 2-class problems with sigmoid output |
| **RMSprop** | Adaptive gradient optimiser; normalises updates by running squared gradient mean |
| **Batch** | A fixed number of images processed together before one weight update |
| **Epoch** | One complete pass through all training images |
| **Callback** | Function called at end of each epoch; used for checkpointing, LR reduction, early stopping |
| **Feature extraction** | Using conv layers to detect patterns; dense layers to classify |
| **Transfer learning** | Using filters learned on a large dataset (ImageNet) as a starting point |

---

## Acknowledgements

- **Dataset:** Kaggle Dogs vs Cats, curated by Google ML Education  
- **Framework:** TensorFlow / Keras by Google  
- **Inspiration:** François Chollet, "Deep Learning with Python" (Manning, 2021) — the canonical reference for this exact problem

---

*Built as a Computer Vision learning project. Fork, experiment, extend.*
