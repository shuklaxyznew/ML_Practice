"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           CAT vs DOG CLASSIFIER — Full CNN Pipeline                         ║
║           Computer Vision | TensorFlow / Keras                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Steps covered:                                                              ║
║    1. Dataset download & directory setup                                     ║
║    2. Image preprocessing (resize, rescale, colour channels)                 ║
║    3. Data augmentation (flip, rotate, zoom, contrast)                       ║
║    4. Understanding convolution, filters & pooling                           ║
║    5. Building the CNN architecture                                          ║
║    6. Training with callbacks                                                ║
║    7. Evaluation & visualisation                                             ║
║    8. Prediction on new images                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHY THIS PROBLEM IS HARDER THAN MNIST
──────────────────────────────────────
MNIST: 28×28 grayscale, centred digit, white background, zero clutter.
Cat vs Dog: 150×150 (or more) RGB, subject anywhere in frame, varied lighting,
            backgrounds, poses, breeds. The model must learn MUCH richer features.

WHAT A CNN LEARNS LAYER BY LAYER
──────────────────────────────────
  Layer 1  →  edges, colour blobs (horizontal/vertical lines)
  Layer 2  →  corners, curves, textures (fur, whiskers)
  Layer 3  →  parts (ear shape, eye, snout)
  Layer 4  →  whole face / body structure
  Output   →  "this is a cat" / "this is a dog"
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os                          # file & directory operations
import zipfile                     # unzip the downloaded dataset
import urllib.request              # download dataset from the internet

import numpy as np                 # numerical operations on arrays
import matplotlib.pyplot as plt    # plotting images and training curves
import matplotlib.image as mpimg   # reading image files for display

import tensorflow as tf            # deep learning engine
from tensorflow import keras       # high-level API (sits on top of TF)
from tensorflow.keras import layers, models   # building blocks for the CNN
from tensorflow.keras.preprocessing import image as keras_image  # image utilities

# Reproducibility — same random seed = same results every run
tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow version : {tf.__version__}")
print(f"Keras version      : {keras.__version__}")
print(f"GPU available      : {len(tf.config.list_physical_devices('GPU')) > 0}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATASET DOWNLOAD & DIRECTORY SETUP
# ═══════════════════════════════════════════════════════════════════════════════
"""
THEORY
──────
We use the Kaggle Dogs vs Cats dataset (subset).
The full dataset has 25,000 images. We use a 2,000-image subset:
  - 1,000 cats  (800 train + 200 validation)
  - 1,000 dogs  (800 train + 200 validation)

WHY A SUBSET?
  Training on 25k images for a tutorial takes too long on CPU.
  2,000 images teaches all the same concepts in minutes.
  The cost is slightly lower accuracy (~85% vs ~95% with full dataset).

DIRECTORY STRUCTURE EXPECTED BY KERAS ImageDataGenerator
  data/
  ├── train/
  │   ├── cats/    ← all training cat images go here
  │   └── dogs/    ← all training dog images go here
  └── validation/
      ├── cats/    ← all validation cat images go here
      └── dogs/    ← all validation dog images go here

Keras reads the FOLDER NAME as the class label automatically.
So "cats/" folder → label "cats", "dogs/" folder → label "dogs".
No manual labelling needed.
"""

# Base directory for everything related to this project
BASE_DIR = 'data'

# Sub-directories for train and validation splits
TRAIN_DIR      = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')

# Class-specific sub-directories
TRAIN_CATS_DIR      = os.path.join(TRAIN_DIR,      'cats')
TRAIN_DOGS_DIR      = os.path.join(TRAIN_DIR,      'dogs')
VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, 'cats')
VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, 'dogs')

def create_directories():
    """Create all required data directories if they don't already exist."""
    dirs = [TRAIN_CATS_DIR, TRAIN_DOGS_DIR,
            VALIDATION_CATS_DIR, VALIDATION_DOGS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)   # exist_ok=True → no error if already exists
    print("✓ Directory structure created:")
    for d in dirs:
        print(f"    {d}/")

create_directories()


def download_and_prepare_dataset():
    """
    Download the Kaggle cats-vs-dogs subset, unzip it, and copy images into
    the expected folder structure.

    The downloaded zip contains:
        PetImages/
        ├── Cat/   (12,500 images: cat.0.jpg, cat.1.jpg, ...)
        └── Dog/   (12,500 images: dog.0.jpg, dog.1.jpg, ...)

    We copy only the first 800 of each into train/ and the next 200 into
    validation/ — giving us 2,000 images total.
    """
    import shutil    # for copying files between directories

    zip_path = 'cats_and_dogs_filtered.zip'
    url = ('https://storage.googleapis.com/mledu-datasets/'
           'cats_and_dogs_filtered.zip')

    if not os.path.exists(zip_path):
        print("Downloading dataset (~60 MB) ...")
        urllib.request.urlretrieve(url, zip_path)
        print("✓ Download complete.")
    else:
        print("✓ Dataset zip already exists, skipping download.")

    # Only extract if images are not already there
    if len(os.listdir(TRAIN_CATS_DIR)) == 0:
        print("Extracting and organising images ...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall('.')   # extracts to cats_and_dogs_filtered/

        # Source paths inside the extracted folder
        src_base  = 'cats_and_dogs_filtered'
        src_cats  = os.path.join(src_base, 'train', 'cats')
        src_dogs  = os.path.join(src_base, 'train', 'dogs')
        val_cats  = os.path.join(src_base, 'validation', 'cats')
        val_dogs  = os.path.join(src_base, 'validation', 'dogs')

        # Copy from extracted folder to our data/ structure
        for fname in sorted(os.listdir(src_cats))[:1000]:
            shutil.copy(os.path.join(src_cats, fname), TRAIN_CATS_DIR)
        for fname in sorted(os.listdir(src_dogs))[:1000]:
            shutil.copy(os.path.join(src_dogs, fname), TRAIN_DOGS_DIR)
        for fname in sorted(os.listdir(val_cats))[:500]:
            shutil.copy(os.path.join(val_cats, fname), VALIDATION_CATS_DIR)
        for fname in sorted(os.listdir(val_dogs))[:500]:
            shutil.copy(os.path.join(val_dogs, fname), VALIDATION_DOGS_DIR)

        print("✓ Images organised.")
    else:
        print("✓ Images already in place, skipping extraction.")

    # Count images in each folder and report
    print("\n── Image counts ──────────────────────────────")
    print(f"  Train cats      : {len(os.listdir(TRAIN_CATS_DIR)):>5}")
    print(f"  Train dogs      : {len(os.listdir(TRAIN_DOGS_DIR)):>5}")
    print(f"  Validation cats : {len(os.listdir(VALIDATION_CATS_DIR)):>5}")
    print(f"  Validation dogs : {len(os.listdir(VALIDATION_DOGS_DIR)):>5}")
    total = (len(os.listdir(TRAIN_CATS_DIR))      +
             len(os.listdir(TRAIN_DOGS_DIR))       +
             len(os.listdir(VALIDATION_CATS_DIR))  +
             len(os.listdir(VALIDATION_DOGS_DIR)))
    print(f"  Total           : {total:>5}")

download_and_prepare_dataset()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
"""
THEORY
──────
Raw images from a camera or dataset are NOT ready for a neural network.
Three problems to fix:

  Problem 1 — SIZE MISMATCH
    Cat photos can be 3000×2000, dog photos 640×480, another 1200×900.
    A neural network needs FIXED-SIZE inputs because the weight matrices
    have fixed dimensions. Solution: resize every image to 150×150 pixels.

  Problem 2 — PIXEL SCALE
    Raw pixel values are integers in [0, 255].
    Large input values → large gradients → unstable training.
    Solution: divide by 255.0 to normalise to [0.0, 1.0].

  Problem 3 — COLOUR CHANNELS
    A colour image has 3 channels: Red, Green, Blue.
    Shape is (height, width, 3).
    The "3" means each pixel has THREE values — one per channel.
    We keep all 3 channels because colour IS useful (cats are often grey,
    golden retrievers are golden — colour helps the classifier).

MATHEMATICS
───────────
Raw image:   X ∈ ℤ^(H × W × 3)   where values ∈ [0, 255]
After resize: X ∈ ℤ^(150 × 150 × 3)
After rescale: X_norm = X / 255.0 ∈ ℝ^(150 × 150 × 3)   values ∈ [0.0, 1.0]

Total input neurons (if flattened): 150 × 150 × 3 = 67,500
  — This is why we use CNNs not MLPs for images.
    A dense layer connecting 67,500 inputs to 128 neurons alone would need
    67,500 × 128 = 8,640,000 parameters just for one layer!
    CNNs use weight sharing (same filter slides everywhere) to be far more
    efficient.

IMPLEMENTATION
──────────────
Keras ImageDataGenerator:
  - Reads images from disk in batches (doesn't load all 2000 at once — saves RAM)
  - Resizes them on-the-fly using target_size
  - Rescales pixel values using rescale=1./255
  - Returns (batch_of_images, batch_of_labels) tuples automatically
"""

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMG_HEIGHT  = 150    # pixels — height to resize all images to
IMG_WIDTH   = 150    # pixels — width  to resize all images to
BATCH_SIZE  = 32     # number of images processed per gradient update
                     # 32 is a common default: fits in RAM, good gradient estimate

# ── Validation generator (NO augmentation — we want real, unmodified images) ──
# rescale=1./255 divides every pixel by 255 → normalises to [0, 1]
validation_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,             # root folder with cats/ and dogs/ subfolders
    target_size=(IMG_HEIGHT, IMG_WIDTH),   # resize every image to 150×150
    batch_size=BATCH_SIZE,
    class_mode='binary'         # binary = two classes (0=cats, 1=dogs)
                                # use 'categorical' if you had 3+ classes
)

print(f"\nClass indices: {validation_generator.class_indices}")
# Output: {'cats': 0, 'dogs': 1}  — Keras assigns labels alphabetically


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
"""
THEORY
──────
We only have 2,000 training images. That's small for a CNN.
The model might memorise the 2,000 images instead of learning general patterns
— this is OVERFITTING.

DATA AUGMENTATION artificially multiplies your dataset by creating slightly
modified versions of existing images ON-THE-FLY during training.

Think of it this way: if you've only ever seen cats facing left, you might
not recognise a cat facing right. Augmentation teaches the model that a
flipped cat is still a cat.

AUGMENTATIONS USED
────────────────────
  1. horizontal_flip  → mirror the image left-right
                         A cat is still a cat when mirrored.

  2. rotation_range=40 → rotate up to 40 degrees randomly
                         A tilted dog is still a dog.

  3. width_shift_range=0.2  → shift image horizontally by up to 20%
  4. height_shift_range=0.2 → shift image vertically by up to 20%
                         The subject isn't always perfectly centred.

  5. shear_range=0.2   → shear transformation (slant the image)
                         Teaches geometric invariance.

  6. zoom_range=0.2    → zoom in or out by up to 20%
                         Cats appear at different distances.

  7. fill_mode='nearest' → when pixels are shifted/rotated, fill empty space
                            by repeating the nearest pixel value.

MATHEMATICS
───────────
Each augmentation is a geometric transformation applied to the pixel grid.

Horizontal flip:
  X_flipped[i, j] = X[i, W-1-j]   (mirror column index)

Rotation by angle θ:
  [x']   [cos θ  -sin θ] [x - cx]   [cx]
  [y'] = [sin θ   cos θ] [y - cy] + [cy]
  where (cx, cy) is the image centre

Zoom (scale factor s):
  X_zoomed[i, j] = X[i/s, j/s]   (sample from scaled coordinates)

KEY INSIGHT: Augmentations are applied RANDOMLY each time an image is loaded.
So the model sees a slightly different version of the same image every epoch.
With 1,000 cat images and 10 epochs → the model sees ~10,000 different cat
variations, dramatically reducing overfitting.

IMPORTANT: Augmentation is applied ONLY to training data, NEVER to validation.
Validation must use real, unmodified images to measure true performance.
"""

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,              # normalise pixel values to [0, 1]

    # ── Geometric augmentations ──────────────────────────────────────────────
    rotation_range=40,           # randomly rotate images 0–40 degrees
    width_shift_range=0.2,       # randomly shift images horizontally (fraction of width)
    height_shift_range=0.2,      # randomly shift images vertically (fraction of height)
    shear_range=0.2,             # randomly apply shearing transformations
    zoom_range=0.2,              # randomly zoom inside pictures
    horizontal_flip=True,        # randomly flip images horizontally

    # ── Fill strategy ────────────────────────────────────────────────────────
    fill_mode='nearest'          # fill newly created pixels after rotation/shift
                                 # 'nearest' = copy from the closest existing pixel
                                 # alternatives: 'constant', 'reflect', 'wrap'
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ── Visualise augmentations ────────────────────────────────────────────────────
# Show the same image with 6 different random augmentations applied to it.
# This helps you verify the augmentations look realistic.
def show_augmentations(img_path):
    """
    Take one image and display 6 augmented versions of it side-by-side.
    This lets you visually verify augmentations are not too extreme.
    """
    img       = keras_image.load_img(img_path)   # load image from disk
    img_array = keras_image.img_to_array(img)    # convert to numpy array (H,W,3)
    img_array = img_array.reshape((1,) + img_array.shape)  # add batch dim → (1,H,W,3)

    # aug_datagen without rescale so we can see RGB colours properly
    aug_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
        fill_mode='nearest'
    )

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    fig.suptitle(f'6 random augmentations of the same image\n{img_path}', fontsize=11)

    for i, batch in enumerate(aug_datagen.flow(img_array, batch_size=1)):
        axes[i].imshow(batch[0].astype('uint8'))  # uint8 for display
        axes[i].axis('off')
        axes[i].set_title(f'Augmentation {i+1}', fontsize=9)
        if i == 5:   # stop after 6 versions
            break

    plt.tight_layout()
    plt.show()

# Show augmentation on the first available cat image
first_cat = os.path.join(TRAIN_CATS_DIR, os.listdir(TRAIN_CATS_DIR)[0])
show_augmentations(first_cat)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CONVOLUTION, FILTERS & POOLING — DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
"""
THEORY: CONVOLUTION
────────────────────
A convolution is a sliding dot-product between a small matrix (the FILTER or
KERNEL) and a patch of the image the same size.

Imagine you have a 5×5 image and a 3×3 filter.
The filter slides across every 3×3 region of the image, one step at a time.
At each position it:
  1. Multiplies each filter value with the corresponding image pixel
  2. Sums all 9 products
  3. Writes the sum to the output (called the FEATURE MAP or ACTIVATION MAP)

MATHEMATICS: SINGLE CONVOLUTION STEP
──────────────────────────────────────
Image patch P (3×3):
  [ 10  20  30 ]
  [ 40  50  60 ]
  [ 70  80  90 ]

Filter W (3×3) — a "vertical edge detector":
  [-1   0   1 ]
  [-1   0   1 ]
  [-1   0   1 ]

Output value at this position:
  Z = Σ (P ⊙ W)   ← element-wise multiply then sum
  Z = (10×-1) + (20×0) + (30×1)
    + (40×-1) + (50×0) + (60×1)
    + (70×-1) + (80×0) + (90×1)
  Z = -10+0+30 - 40+0+60 - 70+0+90
  Z = 60   ← strong positive response → vertical edge detected here!

The filter is then LEARNED during training.
Initially random, gradually adjusted by backpropagation to detect whatever
patterns are most useful for distinguishing cats from dogs.

WHAT DIFFERENT FILTERS DETECT
───────────────────────────────
  Vertical edge filter:
    [-1  0  1]    Fires strongly on vertical colour transitions
    [-1  0  1]
    [-1  0  1]

  Horizontal edge filter:
    [-1 -1 -1]    Fires strongly on horizontal colour transitions
    [ 0  0  0]
    [ 1  1  1]

  Blur filter:
    [1/9 1/9 1/9]   Averages neighbouring pixels → smoothing
    [1/9 1/9 1/9]
    [1/9 1/9 1/9]

  Sharpen filter:
    [ 0 -1  0]    Enhances edges by subtracting neighbours
    [-1  5 -1]
    [ 0 -1  0]

In a CNN, we don't hand-design these filters. We start with random values
and gradient descent learns the optimal filter values automatically.

FEATURE MAP SIZE
────────────────
Input:  H × W,   Filter: F × F,   Padding: P,   Stride: S

Output height = (H - F + 2P) / S + 1
Output width  = (W - F + 2P) / S + 1

With padding='same' (P = F//2):
  Output height = H / S    (same spatial size when S=1)

With padding='valid' (P = 0):
  Output height = (H - F) / S + 1    (shrinks by F-1 pixels)

MULTIPLE FILTERS
────────────────
Conv2D(32, 3×3) means 32 different 3×3 filters.
Each filter produces its own feature map.
The output has shape (H, W, 32) — a stack of 32 feature maps.
Each filter learns to detect a different pattern.

THEORY: POOLING
────────────────
After convolution, feature maps are still large.
MaxPooling2D(2,2) reduces spatial dimensions by 2× by keeping only the
MAXIMUM value in each 2×2 window.

WHY MAX? Because the maximum value represents "was this feature present here?".
We care about WHETHER a feature was detected, not exactly WHERE in the 2×2 block.

This gives the model TRANSLATION INVARIANCE:
  A cat's ear a few pixels up or left → still detected as a cat's ear.

MATHEMATICS: MAXPOOLING
────────────────────────
Feature map patch (2×2):
  [0.8  0.2]
  [0.1  0.9]

MaxPool output: max(0.8, 0.2, 0.1, 0.9) = 0.9

Shape transformation:
  Input:  (150, 150, 32)
  After MaxPool(2,2): (75, 75, 32)   — spatial dims halved, channels unchanged
"""

def visualise_conv_filters_and_maps(model, sample_img_path):
    """
    After a model is trained, this function visualises:
      1. The learned filter weights from the first Conv2D layer
      2. The feature maps (activations) produced by each filter
         when a cat or dog image passes through the first layer.

    This is a KEY debugging tool — it shows you what the CNN 'sees'.
    """
    # ── Part 1: Visualise learned filter weights ──────────────────────────────
    # Get the first Conv2D layer
    conv_layer   = model.layers[0]        # first layer
    filter_weights = conv_layer.get_weights()[0]   # shape: (3, 3, 3, 32)
    # Dimensions: (kernel_H, kernel_W, input_channels, num_filters)
    # [3, 3, 3, 32] = 3×3 filter, 3 input channels (RGB), 32 filters

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('Learned filter weights — first Conv2D layer\n'
                 '(each 3×3 square is one filter, visualised as a tiny image)',
                 fontsize=12)

    for i, ax in enumerate(axes.flat):
        if i < filter_weights.shape[-1]:   # number of filters
            # Take the filter, normalise to [0,1] for display
            f = filter_weights[:, :, :, i]          # shape (3, 3, 3)
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f)
            ax.set_title(f'Filter {i+1}', fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # ── Part 2: Visualise feature maps ───────────────────────────────────────
    # Build a sub-model that outputs just the first Conv2D layer's activations
    feature_map_model = models.Model(
        inputs  = model.input,
        outputs = model.layers[0].output   # output of first conv layer
    )

    # Load and preprocess one image
    img       = keras_image.load_img(sample_img_path,
                                      target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = keras_image.img_to_array(img) / 255.0   # normalise
    img_array = np.expand_dims(img_array, axis=0)        # add batch dim

    # Run the image through just the first layer
    feature_maps = feature_map_model.predict(img_array, verbose=0)
    # feature_maps shape: (1, 150, 150, 32) — 32 feature maps of 150×150 each

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle(f'Feature maps from first Conv2D layer\n'
                 f'(what each of the 32 filters "sees" in the image)',
                 fontsize=12)

    # Show the original image in the first position
    axes[0][0].imshow(img)
    axes[0][0].set_title('Original', fontsize=8)
    axes[0][0].axis('off')

    # Show each feature map
    for i in range(1, 32):
        row, col = divmod(i, 8)
        axes[row][col].imshow(feature_maps[0, :, :, i-1], cmap='viridis')
        axes[row][col].set_title(f'Filter {i}', fontsize=7)
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — CNN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
"""
ARCHITECTURE DESIGN PHILOSOPHY
────────────────────────────────
We use a PROGRESSIVE DEEPENING strategy:
  - Start with few filters (32) to detect simple features (edges)
  - Double filters at each block (32 → 64 → 128) as we go deeper
  - Each block = Conv2D → ReLU → MaxPool
  - Deeper layers detect complex patterns by combining simple ones

The intuition:
  Block 1 (32 filters): "I see a curved line here, a straight edge there"
  Block 2 (64 filters): "Those edges form a pointy ear shape"
  Block 3 (128 filters): "That ear + that flat nose = cat face"
  Dense layer: "High cat score → output: cat"

COMPLETE ARCHITECTURE
──────────────────────
  Input:        (150, 150, 3)    ← RGB image
  ─────────────────────────────────────────────────────
  Conv2D(32, 3×3, relu)         → (150, 150, 32)
  MaxPooling2D(2×2)             → ( 75,  75, 32)
  ─────────────────────────────────────────────────────
  Conv2D(64, 3×3, relu)         → ( 75,  75, 64)
  MaxPooling2D(2×2)             → ( 37,  37, 64)
  ─────────────────────────────────────────────────────
  Conv2D(128, 3×3, relu)        → ( 37,  37, 128)
  MaxPooling2D(2×2)             → ( 18,  18, 128)
  ─────────────────────────────────────────────────────
  Conv2D(128, 3×3, relu)        → ( 18,  18, 128)
  MaxPooling2D(2×2)             → (  9,   9, 128)
  ─────────────────────────────────────────────────────
  Flatten                        → (10368,)
  Dropout(0.5)                   → (10368,)     ← regularisation
  Dense(512, relu)               → (512,)
  Dense(1, sigmoid)              → (1,)         ← probability cat=0 / dog=1

OUTPUT ACTIVATION: SIGMOID (not softmax!)
──────────────────────────────────────────
For BINARY classification (2 classes), we use:
  - 1 output neuron
  - Sigmoid activation: σ(x) = 1 / (1 + e^{-x}) → output ∈ (0, 1)
  - Interpret: output = P(dog)
    If output > 0.5 → predict dog
    If output < 0.5 → predict cat
    If output = 0.3 → "30% dog, 70% cat" → predict cat

Loss function for binary classification:
  Binary Cross-Entropy:
  L = −(1/N) Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]

  where y ∈ {0,1} is the true label and ŷ ∈ (0,1) is the predicted probability.

DROPOUT
────────
Dropout(0.5) randomly sets 50% of neurons to 0 during each training forward pass.
This forces the network to not rely on any single neuron — it must learn redundant
representations. At test/predict time, Dropout is DISABLED (all neurons active,
but outputs scaled by 0.5 to compensate).
"""

def build_cat_dog_cnn():
    """
    Builds and returns the Cat vs Dog CNN.

    Architecture: 4 × (Conv2D + MaxPool) blocks → Dropout → Dense(512) → Dense(1)
    Input shape : (150, 150, 3)   — RGB images
    Output      : scalar in (0,1) — probability of being a dog
    """
    model = models.Sequential([

        # ── Convolutional Block 1 ────────────────────────────────────────────
        # 32 filters, each 3×3. Detects basic edges and colour gradients.
        # padding='valid' (default): output shrinks by 2 pixels each side
        # activation='relu': applies max(0,x) after each convolution
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),   # (150, 150, 3)
            name='conv_1'
        ),
        # MaxPool halves spatial dimensions: 148×148 → 74×74
        layers.MaxPooling2D(pool_size=(2, 2), name='pool_1'),

        # ── Convolutional Block 2 ────────────────────────────────────────────
        # 64 filters — more filters = more distinct patterns to detect.
        # Layer 2 combines patterns from layer 1 → detects textures, simple shapes.
        layers.Conv2D(64, (3, 3), activation='relu', name='conv_2'),
        layers.MaxPooling2D((2, 2), name='pool_2'),

        # ── Convolutional Block 3 ────────────────────────────────────────────
        # 128 filters — detects complex patterns (curves, object parts).
        layers.Conv2D(128, (3, 3), activation='relu', name='conv_3'),
        layers.MaxPooling2D((2, 2), name='pool_3'),

        # ── Convolutional Block 4 ────────────────────────────────────────────
        # Another 128-filter block for even deeper feature extraction.
        # By this point the spatial size is very small (9×9) but very information-rich.
        layers.Conv2D(128, (3, 3), activation='relu', name='conv_4'),
        layers.MaxPooling2D((2, 2), name='pool_4'),

        # ── Transition: 3D → 1D ──────────────────────────────────────────────
        # Flatten converts the 3D tensor (9, 9, 128) into a 1D vector (10368,).
        # This is necessary before Dense layers, which expect 1D input.
        layers.Flatten(name='flatten'),

        # ── Regularisation ───────────────────────────────────────────────────
        # Dropout(0.5): randomly zero 50% of activations during training.
        # Prevents the dense layers from memorising training data.
        # CRITICAL: Dropout is active only during training (model.fit),
        #           NOT during evaluation (model.evaluate, model.predict).
        layers.Dropout(0.5, name='dropout'),

        # ── Classification Head ───────────────────────────────────────────────
        # Dense(512): fully connected layer combining all extracted features.
        # 512 neurons can represent complex combinations of visual features.
        layers.Dense(512, activation='relu', name='dense_1'),

        # ── Output Layer ─────────────────────────────────────────────────────
        # Dense(1, sigmoid): single neuron with sigmoid activation.
        # sigmoid(x) = 1 / (1 + e^{-x}) → output ∈ (0, 1)
        # Interpret: output close to 1 → dog, output close to 0 → cat
        # Use Dense(1, sigmoid) + binary_crossentropy for TWO-class problems.
        # Use Dense(N, softmax) + categorical_crossentropy for N-class problems.
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    return model


model = build_cat_dog_cnn()
model.summary()

# Count parameters manually to understand model size
total_params     = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\nTotal parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print(f"Memory (float32)     : ~{total_params * 4 / 1024:.0f} KB")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — COMPILE & TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
"""
COMPILE
────────
Before training we must configure three things:

  1. OPTIMIZER — how to update weights
     RMSprop: similar to Adam, good for CNNs on image data.
     Maintains a moving average of squared gradients to normalise updates.

     RMSprop update rule:
       v_t = ρ · v_{t-1} + (1−ρ) · g_t²          ← running avg of squared grad
       W_t = W_{t-1} − (α / √(v_t + ε)) · g_t    ← normalised update

     Defaults: learning_rate=1e-4, ρ=0.9

  2. LOSS FUNCTION — what to minimise
     binary_crossentropy for 2-class problems:
       L = −(1/N) Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]
     Good behaviour:
       Perfect prediction (ŷ=1, y=1): L = −log(1) = 0  ← zero loss ✓
       Terrible prediction (ŷ≈0, y=1): L = −log(ε) → ∞ ← huge loss ✓

  3. METRICS — what to report (does NOT affect training)
     accuracy = fraction of correct predictions

TRAINING
─────────
model.fit() runs the training loop:
  FOR each epoch:
    FOR each batch of 32 images:
      1. Forward pass → compute predictions
      2. Compute binary_crossentropy loss
      3. Backward pass → compute gradients via chain rule
      4. RMSprop → update all weights
    Report epoch-level metrics (train loss, train acc, val loss, val acc)
"""

model.compile(
    # RMSprop with a SMALL learning rate (1e-4).
    # Why small? Images are complex. Large learning rate → overshoot optimal weights.
    # Rule of thumb: start with 1e-3, reduce to 1e-4 if training is unstable.
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),

    # binary_crossentropy: correct loss for 2-class problems with sigmoid output.
    # from_logits=False (default) because our output layer uses sigmoid (not raw logits).
    loss='binary_crossentropy',

    # Track accuracy — easy to interpret, but note: accuracy can be misleading
    # for imbalanced datasets (if 90% are cats, predicting cat always = 90% acc).
    # Our dataset is balanced so accuracy is reliable here.
    metrics=['accuracy']
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = 'best_cat_dog_model.keras'

callbacks = [
    # Save the model whenever validation accuracy improves.
    # 'save_best_only=True': only keeps the single best version, not every epoch.
    keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    # Reduce learning rate when training stagnates.
    # If val_loss doesn't improve for 5 epochs → multiply lr by 0.5.
    # Helps escape plateaus in the loss landscape.
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,       # new_lr = current_lr × 0.5
        patience=5,       # wait 5 epochs before reducing
        min_lr=1e-7,      # never go below this
        verbose=1
    ),

    # Stop training early if val_accuracy hasn't improved in 10 epochs.
    # Prevents wasting time and prevents extreme overfitting.
    # restore_best_weights=True: when training stops, reload the best checkpoint.
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

# ── Run training ──────────────────────────────────────────────────────────────
NUM_EPOCHS = 30   # maximum epochs — EarlyStopping will likely stop before this

# steps_per_epoch = total training images / batch size
# With 2000 train images and batch=32: 2000/32 = 62 steps per epoch
# Each step = one batch of 32 images → one weight update
STEPS_PER_EPOCH       = train_generator.samples      // BATCH_SIZE
VALIDATION_STEPS      = validation_generator.samples // BATCH_SIZE

print(f"\nTraining configuration:")
print(f"  Training images   : {train_generator.samples:,}")
print(f"  Validation images : {validation_generator.samples:,}")
print(f"  Batch size        : {BATCH_SIZE}")
print(f"  Steps per epoch   : {STEPS_PER_EPOCH}")
print(f"  Max epochs        : {NUM_EPOCHS}")
print(f"\nStarting training...\n")

history = model.fit(
    train_generator,                    # yields batches of (images, labels)
    steps_per_epoch=STEPS_PER_EPOCH,   # how many batches = 1 epoch
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=1
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — EVALUATION & VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history):
    """
    Plot training and validation accuracy + loss curves.

    READING THE CURVES:
    ────────────────────
    Good training:
      train_acc rises, val_acc rises together, both plateau at high value.
      train_loss falls, val_loss falls together.

    Overfitting:
      train_acc >> val_acc (model memorised training data).
      val_loss RISES while train_loss falls.
      FIX: more augmentation, more dropout, fewer parameters.

    Underfitting:
      Both accuracies plateau too low (< 75%).
      FIX: more capacity (more filters/layers), more epochs, less dropout.
    """
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax1.plot(epochs, acc,     'b-o', markersize=4, label='Training accuracy')
    ax1.plot(epochs, val_acc, 'r-o', markersize=4, label='Validation accuracy')
    ax1.axhline(y=max(val_acc), color='gray', linestyle='--', alpha=0.5,
                label=f'Best val acc: {max(val_acc):.3f}')
    ax1.set_title('Training and Validation Accuracy', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax2.plot(epochs, loss,     'b-o', markersize=4, label='Training loss')
    ax2.plot(epochs, val_loss, 'r-o', markersize=4, label='Validation loss')
    ax2.axhline(y=min(val_loss), color='gray', linestyle='--', alpha=0.5,
                label=f'Best val loss: {min(val_loss):.3f}')
    ax2.set_title('Training and Validation Loss', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Binary cross-entropy loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Model Training History', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.show()

    print(f"\n── Final Results ─────────────────────────────────")
    print(f"  Best validation accuracy : {max(val_acc):.4f}  ({max(val_acc)*100:.1f}%)")
    print(f"  Best validation loss     : {min(val_loss):.4f}")
    print(f"  Stopped at epoch         : {len(acc)}")

plot_training_history(history)

# ── Evaluate on validation set ─────────────────────────────────────────────────
model.load_weights(CHECKPOINT_PATH)   # reload best checkpoint
val_loss_final, val_acc_final = model.evaluate(
    validation_generator, steps=VALIDATION_STEPS, verbose=0
)
print(f"\nFinal evaluation on validation set:")
print(f"  Accuracy : {val_acc_final:.4f}  ({val_acc_final*100:.1f}%)")
print(f"  Loss     : {val_loss_final:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — VISUALISE WHAT THE CNN LEARNED
# ═══════════════════════════════════════════════════════════════════════════════

# Call the visualisation function defined in Step 4
# This shows learned filter weights and feature maps
sample_cat_path = os.path.join(TRAIN_CATS_DIR, os.listdir(TRAIN_CATS_DIR)[0])
visualise_conv_filters_and_maps(model, sample_cat_path)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — PREDICTION ON NEW IMAGES
# ═══════════════════════════════════════════════════════════════════════════════
"""
THEORY: INFERENCE
──────────────────
At prediction time:
  1. Load the image and resize to 150×150
  2. Normalise pixels to [0, 1]
  3. Add batch dimension: (150,150,3) → (1,150,150,3)
  4. model.predict() does a forward pass (Dropout disabled)
  5. Output is a single float in (0, 1)
     < 0.5 → cat,  ≥ 0.5 → dog

MATHEMATICS
────────────
  output = σ(z_final) where z_final is the last layer's raw score
  σ(x) = 1 / (1 + e^{-x})

  P(dog)  = output
  P(cat)  = 1 - output

  Confidence of cat prediction: (1 - output) × 100%
  Confidence of dog prediction: output × 100%
"""

def predict_image(img_path, model):
    """
    Takes a path to any cat or dog image and returns the model's prediction.

    Parameters
    ----------
    img_path : str   path to the image file
    model    : keras.Model   the trained classifier

    Returns
    -------
    dict with keys: 'prediction', 'confidence', 'raw_output'
    """
    # Step 1: Load image and resize to the size the model expects
    img = keras_image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

    # Step 2: Convert to numpy array — shape becomes (150, 150, 3)
    img_array = keras_image.img_to_array(img)

    # Step 3: Normalise pixel values from [0,255] to [0.0, 1.0]
    img_array = img_array / 255.0

    # Step 4: Add batch dimension — CNN expects (batch, H, W, C)
    # np.expand_dims adds a size-1 dimension at position 0
    # (150, 150, 3) → (1, 150, 150, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # Step 5: Forward pass — Dropout is automatically disabled during predict()
    raw_output = model.predict(img_batch, verbose=0)[0][0]
    # raw_output is a single float, e.g. 0.87

    # Step 6: Interpret the output
    if raw_output >= 0.5:
        prediction = 'DOG'
        confidence = raw_output * 100           # e.g. 87%
    else:
        prediction = 'CAT'
        confidence = (1 - raw_output) * 100    # e.g. 92%

    return {
        'prediction' : prediction,
        'confidence' : confidence,
        'raw_output' : raw_output
    }


def predict_and_display(img_paths, model):
    """Display a grid of images with their predictions."""
    n = len(img_paths)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for i, img_path in enumerate(img_paths):
        row, col = divmod(i, cols)
        result = predict_image(img_path, model)

        img = keras_image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        axes[row][col].imshow(img)
        axes[row][col].set_title(
            f"{result['prediction']}  ({result['confidence']:.1f}% confident)\n"
            f"Raw sigmoid: {result['raw_output']:.3f}",
            fontsize=10,
            color='royalblue' if result['prediction'] == 'DOG' else 'darkorange'
        )
        axes[row][col].axis('off')

    # Hide any unused subplots
    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        axes[row][col].axis('off')

    plt.suptitle('Cat vs Dog Predictions', fontsize=14)
    plt.tight_layout()
    plt.show()


# Predict on a sample of validation images
sample_images = (
    [os.path.join(VALIDATION_CATS_DIR, f)
     for f in os.listdir(VALIDATION_CATS_DIR)[:4]] +
    [os.path.join(VALIDATION_DOGS_DIR, f)
     for f in os.listdir(VALIDATION_DOGS_DIR)[:4]]
)
predict_and_display(sample_images, model)


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE THE FINAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

model.save('cat_dog_cnn_final.keras')
print("\n✓ Model saved to: cat_dog_cnn_final.keras")
print("  To reload: model = tf.keras.models.load_model('cat_dog_cnn_final.keras')")
print("\n── Training complete ──────────────────────────────────")
