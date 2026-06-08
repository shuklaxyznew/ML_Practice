# CLIP From First Principles

**A complete, self-contained teaching repository for Contrastive Language-Image Pre-Training (CLIP).**

Covers every concept from the ground up: the intuition, the mathematics, runnable experiments
(NumPy only — no GPU required), and an optional mode that runs against the actual OpenAI model weights.

---

## Table of Contents

1. [What Is CLIP?](#1-what-is-clip)
2. [Repository Structure](#2-repository-structure)
3. [Quick Start](#3-quick-start)
4. [Dependencies](#4-dependencies)
5. [Concept Guide](#5-concept-guide)
   - 5.1 [Contrastive Learning](#51-contrastive-learning)
   - 5.2 [Image Encoder (ViT)](#52-image-encoder-vit)
   - 5.3 [Text Encoder (GPT-style)](#53-text-encoder-gpt-style)
   - 5.4 [Shared Embedding Space](#54-shared-embedding-space)
   - 5.5 [Zero-Shot Classification](#55-zero-shot-classification)
6. [Experiment Reference](#6-experiment-reference)
   - Exp 1: [Contrastive Loss on a Batch](#experiment-1--contrastive-loss-on-a-batch)
   - Exp 2: [Zero-Shot Classification](#experiment-2--zero-shot-classification)
   - Exp 3: [Temperature τ Ablation](#experiment-3--temperature-τ-ablation)
   - Exp 4: [Embedding Geometry](#experiment-4--embedding-geometry)
   - Exp 5: [Real CLIP (optional)](#experiment-5--real-clip-optional)
7. [Mathematical Reference](#7-mathematical-reference)
8. [Architecture Details](#8-architecture-details)
9. [Training Details](#9-training-details)
10. [Key Results from the Paper](#10-key-results-from-the-paper)
11. [Limitations and Failure Modes](#11-limitations-and-failure-modes)
12. [Extensions and Further Reading](#12-extensions-and-further-reading)
13. [Glossary](#13-glossary)
14. [Citation](#14-citation)

---

## 1. What Is CLIP?

**CLIP** (Contrastive Language-Image Pre-Training) is a neural network model introduced by OpenAI
in 2021 that learns visual concepts from natural language descriptions.

### The core insight

Traditional computer vision models are trained with *labelled categories*:
you show the model a million dog photos each tagged "dog", a million cat photos tagged "cat", and so on.
This is expensive (human labellers), brittle (fixed category set), and wasteful (ignores the rich
language in image captions).

CLIP instead learns from *image-text pairs* scraped from the internet:

```
Image: [photo of a golden retriever]  ←→  Text: "A fluffy dog playing in the park"
Image: [photo of a red sports car]    ←→  Text: "Ferrari on the racetrack"
Image: [photo of Mount Fuji]          ←→  Text: "Snow-capped mountain in Japan"
```

No manual labels. No fixed category list. Just the natural language that people write alongside images.

### Why it matters

After training on 400 million such pairs, CLIP can:

- **Classify images into any category** described in plain English — including categories it has never
  explicitly seen during training (zero-shot classification).
- **Search images by text query** — find photos matching "a sunset over the ocean" without any labels.
- **Measure image-text similarity** — score how well a caption describes a photo.
- **Serve as a visual backbone** for downstream tasks with no or very little fine-tuning.

### The landmark result

CLIP's ViT-L/14 model achieves **76.2% top-1 accuracy on ImageNet in the zero-shot setting** —
matching a ResNet-50 that was *trained from scratch on 1.28 million labelled ImageNet images*.
No ImageNet training images were ever shown to CLIP during training.

---

## 2. Repository Structure

```
clip-first-principles/
├── README.md                  ← You are here. Complete documentation.
├── clip_experiment.py         ← All experiments. Run this file.
│
│   Inside clip_experiment.py:
│   ├── l2_norm()              ← L2 normalisation onto unit sphere
│   ├── cosine_sim()           ← Cosine similarity (= dot product on unit sphere)
│   ├── softmax()              ← Numerically stable softmax with temperature
│   ├── cross_entropy()        ← Row-wise cross-entropy loss
│   ├── symmetric_clip_loss()  ← Full CLIP training objective
│   │
│   ├── experiment_contrastive_loss()   ← Exp 1: N×N similarity matrix + loss
│   ├── experiment_zero_shot()          ← Exp 2: classify image with text labels
│   ├── experiment_temperature()        ← Exp 3: τ sharpness ablation
│   ├── experiment_embedding_geometry() ← Exp 4: semantic cluster structure
│   └── run_real_clip()                 ← Exp 5: real model weights (optional)
```

---

## 3. Quick Start

### Simulation mode (no downloads, runs in < 1 second)

```bash
# Clone or download this repository
# Then run:
python clip_experiment.py
```

Expected output (abbreviated):

```
================================================================
EXPERIMENT 1: Contrastive Loss on a Batch of 4 Pairs
================================================================
Cosine similarity matrix (τ=0.07, scaled = S/τ):
                 dog photo   car photo    mountain       pizza
  dog photo         6.46 ✓     -1.13        0.50        0.30
  ...

Symmetric CLIP loss (well-aligned batch): 0.01270
Symmetric CLIP loss (random/untrained):   1.34001
Ratio (untrained / trained): 105.5×
...
```

### Real model mode (downloads ~600 MB on first run)

```bash
pip install torch transformers Pillow requests
python clip_experiment.py --real
```

---

## 4. Dependencies

### Simulation mode (Experiments 1–4)

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.21  | All linear algebra (vectors, matrices, norms, exp) |

That's it. NumPy is included in every standard Python installation.

### Real CLIP mode (Experiment 5)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 1.12  | Tensor operations + model inference |
| `transformers` | ≥ 4.20 | HuggingFace wrapper for CLIP model + processor |
| `Pillow` | ≥ 9.0  | Image loading and pre-processing |
| `requests` | ≥ 2.28 | Downloading the test image from Wikipedia |

Install all at once:
```bash
pip install torch transformers Pillow requests
```

For GPU inference (optional, gives ~10× speedup):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Python version

Tested on Python 3.8, 3.9, 3.10, 3.11, 3.12. Should work on any Python ≥ 3.8.

---

## 5. Concept Guide

### 5.1 Contrastive Learning

#### Intuition

Imagine you are a librarian handed a shuffled pile of 4096 photos and 4096 caption cards. Your job:
match every photo to its correct caption. You can't read the captions first — you have to learn what
makes a photo-caption pair "correct" purely from co-occurrence statistics.

Contrastive learning does exactly this. Given a batch of N (image, text) pairs:

- **Pull together**: the N pairs that actually match (the correct photo ↔ its caption).
- **Push apart**: the N² − N pairs that don't match (every photo ↔ every other caption).

After training on 400 million such pairs, the model learns a deep understanding of visual concepts
purely from language supervision.

#### Mathematics

Given a batch of N matched pairs, define:

- `ê_I^(i)` = L2-normalised image embedding for the i-th image  ∈ ℝ^d
- `ê_T^(j)` = L2-normalised text embedding for the j-th text   ∈ ℝ^d

**Step 1 — Similarity matrix** (N × N):

```
S[i, j] = ê_I^(i) · ê_T^(j)    (cosine similarity, since both are unit vectors)
```

The diagonal `S[i, i]` contains the similarity of each correct pair.
Off-diagonal entries `S[i, j]` (i ≠ j) are the "hard negatives" that must be driven low.

**Step 2 — Scale by temperature**:

```
L[i, j] = S[i, j] / τ
```

Where τ (tau) is a learned positive scalar. Smaller τ → sharper distribution.

**Step 3 — Symmetric cross-entropy loss**:

```
L_img = (1/N) Σ_i [ -L[i,i] + log Σ_j exp(L[i,j]) ]   (image→text direction)
L_txt = (1/N) Σ_j [ -L[j,j] + log Σ_i exp(L[i,j]) ]   (text→image direction)
L     = (L_img + L_txt) / 2                              (symmetric average)
```

**Interpretation of the loss**:

- `L_img`: "Given image i, rank text i above all other N-1 texts." This is N independent
  N-class classification problems, one per image in the batch.
- `L_txt`: "Given text j, rank image j above all other N-1 images." Same structure, transposed.
- The symmetric average enforces both constraints simultaneously.

**Why large batches matter**:

| Batch size N | Negative pairs | Context |
|:---:|:---:|:---|
| 4 | 12 | This README's demo |
| 256 | 65,280 | Small-scale training |
| 4,096 | ~16.7 million | CLIP's actual training (ViT-B) |
| 32,768 | ~1.07 billion | CLIP's training (ViT-L, with multi-GPU sharding) |

More negatives per step → richer gradient signal → better representations.

#### Why it works (informal proof)

The loss is minimised when `S[i,i]` is much larger than every `S[i,j]` (i≠j) in each row.
Gradient descent pushes the encoder weights in the direction that:

1. Maps matching image-text pairs to nearby points on the unit sphere.
2. Maps non-matching pairs to distant points on the unit sphere.

After convergence, the sphere's geometry *is* the model's world knowledge:
nearby points = semantically similar concepts.

---

### 5.2 Image Encoder (ViT)

#### Intuition

CLIP processes images with a **Vision Transformer (ViT)**. Think of it as reading an image the same
way a transformer reads a sentence: chop it into pieces, convert each piece to a vector (a "word"),
then apply self-attention to understand how pieces relate to each other.

```
[224×224 image]
       ↓
   Chop into 14×14 grid of 16×16 patches  =  196 patches
       ↓
   Linear projection: each patch → 768-d token vector
       ↓
   Add positional embedding (so the model knows which patch is where)
       ↓
   Prepend [CLS] token (a learnable vector that will accumulate global info)
       ↓
   12 layers of Multi-Head Self-Attention + Feed-Forward Network
       ↓
   Extract [CLS] token output  →  768-d image representation
       ↓
   Linear projection  →  512-d embedding
       ↓
   L2 normalise  →  ê_I ∈ ℝ^512, ‖ê_I‖₂ = 1
```

#### Mathematics

**Patch embedding**

For an image X ∈ ℝ^(H×W×C) with H=W=224, C=3 (RGB), patch size P=16:

```
Number of patches: N_p = (H/P) × (W/P) = 14 × 14 = 196

Each patch: x_p ∈ ℝ^(P×P×C) = ℝ^768    (16×16×3 pixels flattened)

Patch embedding: e_p = W_E · x_p + b_E    W_E ∈ ℝ^(768×d_model)
```

**Positional encoding**

Unlike CNNs, transformers have no built-in notion of spatial position. CLIP adds a learned
positional embedding to each patch token:

```
z_0 = [ e_cls, e_1 + pos_1, e_2 + pos_2, ..., e_196 + pos_196 ]
```

Where `e_cls` is a learnable "class token" prepended to the sequence (length now 197).

**Self-attention (one layer)**

Each transformer layer applies multi-head self-attention followed by a feed-forward network:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

where:  Q = z · W_Q    (queries)
        K = z · W_K    (keys)
        V = z · W_V    (values)
        d_k = d_model / num_heads    (per-head dimension)
```

In CLIP's ViT-B/32: `d_model=768`, `num_heads=12`, `d_k=64`.

Self-attention is computed across all 197 tokens (196 patches + 1 CLS), so every patch
can "look at" every other patch in a single layer.

**Output extraction**

After L=12 transformer layers, the output corresponding to the `[CLS]` token position is
a 768-d vector that has aggregated information from all 196 patches through attention.

```
f_I = z_L[0]     (first position = CLS token output), f_I ∈ ℝ^768

Project to shared space:  e_I = W_I · f_I    W_I ∈ ℝ^(768×512)

Normalise:  ê_I = e_I / ‖e_I‖₂    ê_I ∈ ℝ^512, ‖ê_I‖₂ = 1
```

#### Model variants

| Model | Patch size | d_model | Layers | Heads | Params | ImageNet 0-shot |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| ViT-B/32 | 32×32 | 768 | 12 | 12 | 88M | 63.3% |
| ViT-B/16 | 16×16 | 768 | 12 | 12 | 86M | 68.3% |
| ViT-L/14 | 14×14 | 1024 | 24 | 16 | 307M | 75.3% |
| ViT-L/14@336 | 14×14 | 1024 | 24 | 16 | 307M | **76.2%** |

Smaller patch = more patches = more computation but better accuracy. ViT-L/14@336 uses a
336×336 input resolution, giving (336/14)² = 576 patches instead of 196.

**CLIP also supports ResNet encoders** (RN50, RN101, RN50x4, RN50x16, RN50x64) as an alternative
to ViT. The attention pooling variant of ResNet performs comparably but ViT-L/14 is the strongest.

---

### 5.3 Text Encoder (GPT-style)

#### Intuition

The text encoder is essentially a GPT-style transformer — identical in architecture to the
language model that powers text generation, but used here for embedding rather than generation.

Given a text string like `"a golden retriever playing fetch"`:

```
Tokenise with BPE → [SOT, "a", "golden", "retriever", "playing", "fetch", EOT]
       ↓
Lookup table: each token ID → 512-d embedding vector
       ↓
Add positional embeddings (learned, up to 77 positions)
       ↓
12 layers of causal (masked) self-attention + FFN
       ↓
Extract the [EOT] token output  →  512-d text representation
       ↓
Linear projection + L2 normalise  →  ê_T ∈ ℝ^512
```

#### Mathematics

**Byte-Pair Encoding (BPE) tokenisation**

CLIP uses a custom BPE vocabulary of 49,152 tokens built from the WIT training corpus.
BPE is a subword tokenisation algorithm: common words become single tokens, rare words
are split into character-level pieces. This makes the vocabulary robust to typos,
compound words, scientific terms, and multilingual text.

```
"golden retriever" → ["golden", "retriev", "er"]   (hypothetical split)
"a"                → ["a"]
"supercalifragilistic" → ["super", "cal", "if", "rag", "ili", "stic"]
```

**Maximum context length**: 77 tokens (including [SOT] and [EOT] markers).
Texts longer than 77 tokens are truncated; shorter texts are padded.

**Causal (autoregressive) masking**

Unlike the ViT's bidirectional attention (each patch sees all other patches), the text
encoder uses a causal mask — each token can only attend to tokens to its left:

```
Mask[i, j] = { 0       if j ≤ i   (attend)
             { -∞      if j > i   (block)
```

This means:
- `"golden"` sees `[SOT, "a", "golden"]` — its own left context
- `[EOT]` sees the entire sequence (all tokens before it)

This is why CLIP uses the `[EOT]` token's output as the text representation — it has
attended to every other token and accumulated the full sentence meaning.

**The causal mask is inherited from GPT architecture, not fundamental to CLIP.**
Ablation studies suggest bidirectional text encoders (BERT-style) perform similarly,
but CLIP used GPT's architecture to leverage existing pre-trained weights during development.

**Output extraction**

```
Let tokens = [t_0, t_1, ..., t_L]   where t_0 = [SOT], t_L = [EOT]

After L=12 transformer layers:
    f_T = h_L    (hidden state at the [EOT] position)
    e_T = W_T · f_T    (project to shared embedding space)
    ê_T = e_T / ‖e_T‖₂    (L2 normalise)
```

---

### 5.4 Shared Embedding Space

#### Intuition

Both encoders project their outputs into the *same* d=512 dimensional vector space.
After training, this space has a rich geometric structure:

- Points near each other are semantically related.
- The geometry is **cross-modal**: a photo of a dog and the text "golden retriever" end up
  as nearby points, even though they went through completely different neural networks.
- The space is **compositional**: "a dog sitting on a red chair" lands between the "dog",
  "sitting", and "red chair" regions.

Think of it as a globe where every concept has a home coordinate. The image encoder and
text encoder both learn to map their respective inputs to the correct address on this globe.

#### Mathematics

**The shared space is the unit hypersphere S^(d-1) in ℝ^d.**

After L2 normalisation, every embedding has length exactly 1. This means all points lie
on the surface of a 511-dimensional sphere embedded in 512-dimensional space.

**Why the unit sphere (not a Euclidean space)?**

1. **Cosine similarity = dot product**: avoids the computational cost of division.
2. **No magnitude bias**: a model can't cheat by making one embedding 100× longer than
   another to dominate the similarity scores.
3. **Well-defined temperature**: the effect of τ is consistent across all training steps
   because the similarity values are always in [-1, +1].

**The similarity matrix in full notation**:

Let:
- `Ê_I ∈ ℝ^(N×d)` = matrix of N image embeddings (each row is a unit vector)
- `Ê_T ∈ ℝ^(N×d)` = matrix of N text embeddings (each row is a unit vector)

```
S = Ê_I @ Ê_T.T    ∈ ℝ^(N×N)
```

This is a single batched matrix multiplication, making it extremely efficient.
On a modern GPU, this takes microseconds even for N=32768.

**The logit matrix**:

```
L = S · exp(log_τ)    where log_τ is the learnable log-temperature
```

CLIP parameterises temperature as `exp(log_τ)` rather than `τ` directly.
This ensures τ stays positive (exp is always positive) and allows larger
gradient steps in log-space.

**Projection layers**:

Both encoders end with a linear layer that maps their final hidden state into the shared
512-d space. These projection matrices `W_I` and `W_T` are also learned during training.

```
e_I = f_I @ W_I    (image encoder output → shared space)
e_T = f_T @ W_T    (text encoder output  → shared space)
```

The projection layers allow the shared space to have a different dimensionality than the
encoder hidden dimensions (e.g., ViT-L has hidden dim 1024 but projects to 768 shared dims).

---

### 5.5 Zero-Shot Classification

#### Intuition

Once the model is trained, zero-shot classification is essentially a nearest-neighbour
search in the shared embedding space. For a new category you've never explicitly trained on:

1. Write the category name as a natural language prompt.
2. Encode it → you get a point on the embedding sphere.
3. Encode the query image → you get another point on the sphere.
4. Check if they're close.

No gradient updates. No labelled examples. Just text.

#### Mathematics

Given:
- A query image I with embedding `ê_I ∈ ℝ^d`
- K candidate class labels `{c_1, c_2, ..., c_K}`

**Step 1**: Form prompts (CLIP's standard template):

```python
prompts = [f"a photo of a {c}" for c in class_labels]
```

**Step 2**: Encode all prompts:

```
Ê_T = [TextEncoder(p_1), TextEncoder(p_2), ..., TextEncoder(p_K)]    ∈ ℝ^(K×d)
```

**Step 3**: Compute similarities:

```
s = ê_I @ Ê_T.T    ∈ ℝ^K    (K cosine similarities, one per class)
```

**Step 4**: Convert to probabilities:

```
P(class_k | image) = exp(s_k / τ) / Σ_j exp(s_j / τ) = softmax(s / τ)[k]
```

**Step 5**: Predict the most likely class:

```
ŷ = argmax_k P(class_k | image)  =  argmax_k s_k    (temperature doesn't change argmax)
```

#### Prompt engineering

The choice of prompt template significantly affects accuracy. CLIP's paper evaluates
several strategies:

| Strategy | ImageNet top-1 accuracy |
|----------|:---:|
| Bare class name: `"dog"` | 73.4% |
| Simple prompt: `"a photo of a dog"` | 74.5% |
| Ensemble of 80 prompts | **76.2%** |

The 80-prompt ensemble includes templates like:
- `"a photo of a {c}"`
- `"a blurry photo of a {c}"`
- `"a black and white photo of the {c}"`
- `"a photo of the large {c}"`
- `"a cropped photo of the {c}"`
- ...

Each prompt is encoded, then the K embedding vectors for each class are **averaged** before
the similarity computation. This is called "prompt ensemble" or "soft prompting".

Why does it work? Because CLIP saw all these kinds of captions during training.
Averaging over prompt variants makes the class representation more robust.

#### Zero-shot on ImageNet: what CLIP is actually matching

When you say "a photo of a dog", CLIP doesn't look for the word "dog" anywhere in its memory.
It encodes the entire *sentence* into a point in embedding space, then checks if the image
embedding is nearby. This works because, during training, the text "golden retriever", "a
playful dog", and "man's best friend" all appear alongside dog photos and cluster near the
same region of the sphere as the dog images.

---

## 6. Experiment Reference

### Experiment 1 — Contrastive Loss on a Batch

**File location**: `experiment_contrastive_loss()` in `clip_experiment.py`

**What it does**:
- Creates N=4 semantic concept vectors in ℝ^512 (dog, car, mountain, pizza)
- Generates simulated image and text embeddings as noisy perturbations of each concept
- Computes the N×N cosine similarity matrix
- Computes and displays the symmetric CLIP loss
- Compares the loss to a random (untrained) batch

**Sample output**:
```
Cosine similarity matrix (τ=0.07, scaled = S/τ):
                 dog photo   car photo    mountain       pizza
  dog photo         6.46 ✓     -1.13        0.50        0.30
  car photo        -1.41        5.66 ✓      0.60        0.55
  mountain          0.30        0.79        5.76 ✓     -0.79
  pizza            -0.48        0.80        0.26        5.35 ✓

Symmetric CLIP loss (well-aligned batch): 0.01270
Symmetric CLIP loss (random/untrained):   1.34001
Ratio (untrained / trained): 105.5×
```

**Key insight**: The diagonal entries (✓) should be the largest in each row and column.
The ratio of ~100× tells us the contrastive loss provides an enormous gradient signal
at the start of training, which is what drives rapid initial learning.

---

### Experiment 2 — Zero-Shot Classification

**File location**: `experiment_zero_shot()` in `clip_experiment.py`

**What it does**:
- Creates 6 concept clusters: dog, car, mountain, pizza, cat, airplane
- Simulates a query dog image (close to the dog concept)
- Computes softmax probabilities over all 6 class labels
- Displays the ranking

**Sample output**:
```
  Class       Similarity  P(class|image)  Bar
  dog            +0.3993          0.9702  ████████████████████████████████
  cat            +0.0482          0.0064
  car            +0.0677          0.0085
  ...
→ Predicted class: "dog" ✓  (confidence: 97.0%)
```

**Key insight**: Even in our simple simulation, the correct class gets 97% of the probability
mass. The temperature τ=0.07 makes the distribution sharp enough that the correct answer
is unambiguous.

---

### Experiment 3 — Temperature τ Ablation

**File location**: `experiment_temperature()` in `clip_experiment.py`

**What it does**:
- Fixes the cosine similarities from a single query
- Applies softmax at 6 different temperature values: 2.0, 0.5, 0.1, 0.07, 0.03, 0.01
- Reports the top-class probability and Shannon entropy at each temperature

**Sample output**:
```
       τ    P_top  H (entropy)  Interpretation
    2.00   0.1898       1.7898  too uncertain — near-uniform distribution
    0.50   0.2728       1.7542  moderate — losing discrimination
    0.10   0.8142       0.7660  CLIP regime — sharp and well-calibrated
    0.07   0.9404       0.3137  CLIP regime — sharp and well-calibrated
    0.03   0.9997       0.0027  overconfident — gradient may vanish
    0.01   1.0000       0.0000  ≈ argmax — no gradient signal
```

**Key insight**: At τ=0.01, the distribution collapses to a hard argmax — the model would
get the correct answer but the gradient would be zero (no learning signal). At τ=2.0,
the model is uncertain enough that it loses discrimination. τ≈0.07 is the Goldilocks value.

---

### Experiment 4 — Embedding Geometry

**File location**: `experiment_embedding_geometry()` in `clip_experiment.py`

**What it does**:
- Creates three semantic category anchors (animal, vehicle, food)
- Builds 8 concept embeddings distributed across the 3 categories
- Queries from a dog image and measures similarities to all 8 concepts
- Shows the within-category vs. between-category distance structure

**Sample output**:
```
  Concept     Category   cos sim  Bar
  dog         animal     +0.7521  ████████████████████████████████████████████████
  cat         animal     +0.1568  ████████████
  bird        animal     +0.0924  ███████
  pasta       food       +0.0343  ███
  car         vehicle    -0.0087  ██
  ...

  dog image → other animals  : +0.1246
  dog image → vehicles       : +0.0004
  dog image → food           : +0.0265
```

**Key insight**: The semantic hierarchy is preserved geometrically. Cat and bird are closer
to the dog image than car or pizza, because they share the "animal" concept anchor. This is
why CLIP can generalise to never-seen categories — the embedding space has generalised
semantic structure, not memorised individual class vectors.

---

### Experiment 5 — Real CLIP (optional)

**File location**: `run_real_clip()` in `clip_experiment.py`

**Requires**:
```bash
pip install torch transformers Pillow requests
python clip_experiment.py --real
```

**What it does**:

**(A) Zero-shot classification on a real dog photo (downloaded from Wikipedia)**
```
  dog         P=0.9421  ████████████████████████████████████████████████
  cat         P=0.0312  █
  bird        P=0.0141
  ...
→ Prediction: "dog"
```

**(B) Prompt engineering ablation**
```
  P(dog)=0.312  prompt: "dog"
  P(dog)=0.441  prompt: "a dog"
  P(dog)=0.793  prompt: "a photo of a dog"
  P(dog)=0.811  prompt: "a high quality photo of a dog"
  P(dog)=0.724  prompt: "a golden retriever dog playing outdoors"
```

**(C) Text-text cosine similarity**
```
  "dog"              ↔ "golden retriever"  :  +0.85  (very high)
  "dog"              ↔ "cat"               :  +0.68  (high — both animals)
  "dog"              ↔ "automobile"        :  +0.24  (low — different domains)
  "machine learning" ↔ "neural network"   :  +0.81  (very high)
  "machine learning" ↔ "cooking recipe"   :  +0.19  (very low)
```

---

## 7. Mathematical Reference

### Complete CLIP loss derivation

**Given**: Batch of N (image, text) pairs: `{(I_1, T_1), ..., (I_N, T_N)}`

**Forward pass**:

```
# Image embeddings (unit vectors in ℝ^d)
ê_I^(i)  =  ImageEncoder(I_i) / ‖ImageEncoder(I_i)‖₂     for i = 1..N

# Text embeddings (unit vectors in ℝ^d)
ê_T^(j)  =  TextEncoder(T_j) / ‖TextEncoder(T_j)‖₂       for j = 1..N

# N×N cosine similarity matrix (single matrix multiply)
S  =  Ê_I @ Ê_T.T    ∈ ℝ^(N×N)
S[i,j]  =  ê_I^(i) · ê_T^(j)    (cosine similarity of pair i,j)

# Scaled logits
L  =  S * exp(log_τ)    (element-wise multiply by the learned inverse temperature)
```

**Loss computation**:

```
# Image→text loss: N-way classification, one per row
# "Given image i, which of the N texts is its correct match?"
L_img  =  (1/N) × Σᵢ CrossEntropy(L[i, :],  target=i)
         =  (1/N) × Σᵢ [ -L[i,i] + log Σⱼ exp(L[i,j]) ]

# Text→image loss: N-way classification, one per column
# "Given text j, which of the N images is its correct match?"
L_txt  =  (1/N) × Σⱼ CrossEntropy(L[:, j],  target=j)
         =  (1/N) × Σⱼ [ -L[j,j] + log Σᵢ exp(L[i,j]) ]

# Symmetric CLIP loss
L  =  (L_img + L_txt) / 2
```

**Gradient direction** (intuition):

The gradient of L with respect to `ê_I^(i)` points in the direction that:
- Increases `S[i,i]` (the correct text gets pulled closer)
- Decreases `S[i,j]` for j≠i (all incorrect texts get pushed further)

The image encoder and text encoder are both updated simultaneously to satisfy this.

### Zero-shot classification formula

```
# K candidate class prompts
prompts  =  ["a photo of a " + c_k  for k in 1..K]

# Encode all K text prompts
Ê_T  =  [TextEncoder(p_k) / ‖TextEncoder(p_k)‖₂  for k in 1..K]   ∈ ℝ^(K×d)

# Single image similarity vector
s  =  ê_I @ Ê_T.T    ∈ ℝ^K

# Class probabilities
P(class_k | I)  =  exp(s_k / τ) / Σⱼ exp(s_j / τ)   =   softmax(s / τ)[k]

# Prediction
ŷ  =  argmax_k  P(class_k | I)  =  argmax_k  s_k
```

---

## 8. Architecture Details

### CLIP ViT-B/32 (the model used in `run_real_clip()`)

```
Image Encoder (ViT-B/32):
  Input size:         224 × 224 × 3
  Patch size:         32 × 32
  Number of patches:  (224/32)² = 7×7 = 49
  +1 [CLS] token:     50 tokens total
  Embedding dim:      768
  Transformer layers: 12
  Attention heads:    12 (per head dim = 64)
  FFN hidden dim:     3072 (= 4 × 768)
  Output dim:         512 (after projection)
  Activation:         QuickGELU (a fast approximation of GELU)
  Normalisation:      Layer norm before each sublayer (Pre-LN)

Text Encoder:
  Vocabulary size:    49,152 (BPE)
  Max context length: 77 tokens
  Embedding dim:      512
  Transformer layers: 12
  Attention heads:    8 (per head dim = 64)
  FFN hidden dim:     2048 (= 4 × 512)
  Attention type:     Causal (autoregressive mask)
  Output dim:         512 (linear projection of [EOT] token)

Shared embedding space:
  Dimension:    d = 512
  Normalisation: L2 (unit hypersphere S^511)
  Temperature:  τ ≈ 0.07 (learned, initialised from log(1/0.07))

Total parameters:    ~150M
```

---

## 9. Training Details

### Dataset: WIT (WebImageText)

- **Size**: 400 million (image, text) pairs
- **Source**: The public internet — images and alt-text collected from web pages
- **Curation**: Basic filtering (English only, minimum text length, deduplication)
- **No manual labels**: The "labels" are whatever text happened to appear alongside the image

### Training hyperparameters (from the paper)

| Hyperparameter | Value |
|---|---|
| Batch size | 32,768 |
| Optimiser | Adam |
| Learning rate (peak) | 5e-4 (with cosine decay) |
| Weight decay | 0.2 |
| β₁, β₂ | 0.9, 0.98 |
| Epsilon | 1e-6 |
| Gradient clip | 1.0 |
| Mixed precision | fp16 |
| Temperature τ init | 0.07 |
| Training duration | 32 epochs (ViT-B/32) to 12 epochs (ViT-L/14) |
| Compute | 256–592 V100 GPUs × 12–18 days |

### Why such a large batch?

With N=32,768 per batch:
- Each step sees 32,768 positive pairs AND ~1.07 billion negative pairs.
- More negatives → richer gradient signal → faster learning of discriminative features.
- The paper shows a near-linear scaling of representation quality with log(batch size).

### The temperature parameter

`log_τ` is initialised to `log(1/0.07) ≈ 2.659` and learned with gradient descent.
During training it typically decreases slightly (τ decreases → sharper distributions).
The paper clips τ to a minimum of 0.01 to prevent training instability.

---

## 10. Key Results from the Paper

### Zero-shot ImageNet performance

| Model | ImageNet Top-1 (0-shot) | Comparison |
|-------|:---:|:---|
| ResNet-50 (supervised, 1.28M labels) | 76.1% | Trained on full ImageNet |
| CLIP ViT-L/14 (zero-shot) | 76.2% | Never saw ImageNet labels |
| CLIP ViT-L/14@336 (zero-shot) | **76.2%** | Higher resolution |
| Human (3 attempts) | ~95% | For reference |

CLIP matches ResNet-50 in zero-shot accuracy despite never being shown a single ImageNet
training label. This was a landmark result in 2021.

### Transfer to other benchmarks (CLIP ViT-L/14)

| Benchmark | Task | CLIP Zero-Shot |
|---|---|:---:|
| ImageNet | Object recognition | 76.2% |
| Oxford Pets | Fine-grained breed classification | 93.5% |
| Stanford Cars | Fine-grained car model classification | 78.8% |
| CIFAR-10 | Object classification (10 classes) | 96.2% |
| CIFAR-100 | Object classification (100 classes) | 80.4% |
| SUN397 | Scene recognition | 68.4% |
| STL-10 | Object recognition | 99.3% |
| Caltech-101 | Object recognition | 87.8% |

### Robustness to distribution shift

One of CLIP's strongest properties is **robustness**. Models trained on ImageNet typically
lose 10–40% accuracy when evaluated on ImageNet-variant datasets (different image styles,
sketches, adversarial examples). CLIP degrades much less:

| Dataset | ResNet-50 | CLIP ViT-B/32 |
|---|:---:|:---:|
| ImageNet | 76.1% | 63.3% |
| ImageNet-V2 | 63.3% | 56.1% |
| ImageNet-Sketch | 26.0% | 46.0% |
| ImageNet-A | 0.0% | 31.3% |
| ObjectNet | 10.0% | 36.0% |

CLIP is worse than ResNet-50 on standard ImageNet but **dramatically better** on distribution-shifted
versions. This is because CLIP learned from text descriptions which generalise better than
pixel statistics.

---

## 11. Limitations and Failure Modes

Understanding where CLIP fails is as important as understanding where it succeeds.

### 1. Abstract and systematic tasks

CLIP struggles with tasks that require counting, spatial reasoning, or systematic comparisons:

```
Query: "a photo of exactly 3 dogs"    → CLIP often returns photos with 1 or 5 dogs
Query: "the cat is to the left of the box"  → spatial relationships are unreliable
```

This is because natural image captions rarely describe exact counts or precise spatial relations.

### 2. Fine-grained discrimination within categories

CLIP is less accurate on fine-grained tasks where all classes are very similar:

| Task | Zero-shot accuracy |
|---|:---:|
| Distinguishing 200 bird species (CUB-200) | ~60% |
| Identifying car models (Stanford Cars) | 78.8% |
| General object recognition (ImageNet 1000 classes) | 76.2% |

Telling a Labrador from a Golden Retriever requires visual detail that is rarely
described in alt-text.

### 3. Novel distributions

CLIP degrades on images that look nothing like natural photographs:

- Medical imagery (X-rays, histology)
- Satellite imagery
- Abstract art
- Mathematical diagrams
- Technical schematics

These domains are underrepresented in the WIT training set.

### 4. Prompt sensitivity

Performance varies significantly with the prompt template, and finding the best prompt
for a given dataset requires manual iteration. This is not ideal for deployment.

### 5. Biases from training data

Because CLIP is trained on internet data, it inherits biases present in that data:
gender, racial, geographic, and cultural biases have been documented in CLIP embeddings.
The embedding space reflects the statistical associations in web text, not ground truth.

### 6. Long text

The 77-token context limit means CLIP cannot encode long descriptions. Everything beyond
77 tokens is ignored. For long captions, only the first 77 tokens matter.

---

## 12. Extensions and Further Reading

### Direct successors

| Model | Year | Key improvement |
|---|---|---|
| ALIGN (Google) | 2021 | Trained on 1.8B noisy pairs; shows scaling laws |
| BASIC | 2022 | 6.6B parameters; 85.7% ImageNet zero-shot |
| BLIP | 2022 | Adds image-text generation; bootstrapped noisy data |
| BLIP-2 | 2023 | Bridges frozen image encoder and LLM with Q-Former |
| SigLIP | 2023 | Sigmoid loss instead of softmax; removes need for large batches |
| OpenCLIP | 2022 | Open-source CLIP reimplementation with scaling study |

### Applications built on CLIP

- **DALL-E 2** (OpenAI): uses CLIP embeddings as the conditioning signal for image generation
- **Stable Diffusion** (Stability AI): uses CLIP text encoder to condition the diffusion U-Net
- **Midjourney**: uses CLIP-related models for text-to-image generation
- **CLIP-Interrogator**: reverse-engineers prompts from images using CLIP similarity
- **CLIP-Seg**: extends CLIP to dense (pixel-level) segmentation with text queries

### Code resources

- **Official CLIP** (OpenAI): `pip install git+https://github.com/openai/CLIP.git`
- **OpenCLIP** (LAION): `pip install open_clip_torch` — open weights, multiple scales
- **HuggingFace Hub**: `CLIPModel.from_pretrained("openai/clip-vit-base-patch32")`

### Papers to read next

1. **CLIP paper** (Radford et al. 2021): https://arxiv.org/abs/2103.00020
2. **ViT paper** (Dosovitskiy et al. 2020): https://arxiv.org/abs/2010.11929
3. **ALIGN** (Jia et al. 2021): https://arxiv.org/abs/2102.05918
4. **Scaling CLIP** (Cherti et al. 2022): https://arxiv.org/abs/2212.07143
5. **SigLIP** (Zhai et al. 2023): https://arxiv.org/abs/2303.15343

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **BPE** | Byte-Pair Encoding. A subword tokenisation algorithm that splits rare words into common subword units. CLIP's vocabulary has 49,152 BPE tokens. |
| **Causal masking** | In the text encoder, each token can only attend to tokens to its left (earlier in the sequence). Prevents information leakage from future tokens. |
| **[CLS] token** | A special learnable token prepended to the image patch sequence. Its final hidden state summarises the entire image (image encoder output). |
| **Contrastive learning** | A self-supervised learning paradigm that trains representations by distinguishing similar (positive) pairs from dissimilar (negative) pairs. |
| **Cosine similarity** | A measure of similarity between two vectors: `cos(a,b) = (a·b)/(‖a‖‖b‖)`. On the unit sphere, equals the dot product. |
| **Embedding space** | The d-dimensional vector space where both image and text representations live after encoding. CLIP uses d=512. |
| **[EOT] token** | End-Of-Text. A special token appended to every text sequence. The hidden state at the EOT position is used as the text embedding. |
| **L2 normalisation** | Dividing a vector by its Euclidean length so the result has length exactly 1. Projects points onto the unit hypersphere. |
| **Logits** | Raw (pre-softmax) scores. In CLIP: the cosine similarities scaled by 1/τ. |
| **Prompt engineering** | Designing the text prompt to maximise task performance. For CLIP: using "a photo of a {class}" instead of just "{class}". |
| **Self-attention** | A mechanism in transformers where each element of a sequence attends to all other elements. Computes weighted averages of value vectors. |
| **Softmax** | A function that converts a vector of scores into a probability distribution (non-negative, sums to 1). |
| **Temperature τ** | A scalar that controls the sharpness of the softmax distribution. Small τ → sharp (confident); large τ → flat (uncertain). Learned in CLIP. |
| **Unit hypersphere** | The set of all vectors in ℝ^d with L2 norm = 1. All CLIP embeddings live here. |
| **ViT** | Vision Transformer. A transformer architecture applied to image patches rather than text tokens. CLIP's image encoder. |
| **WIT** | WebImageText. The 400M (image, text) pair dataset assembled by OpenAI for CLIP training. Not publicly released. |
| **Zero-shot** | Classifying into a category without having seen any labelled examples of that category during training. |

---

## 14. Citation

If you use this code for research or teaching, please cite the original CLIP paper:

```bibtex
@inproceedings{radford2021learning,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya
               and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda
               and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning (ICML)},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.00020}
}
```

---

*README written alongside `clip_experiment.py` as a companion teaching document.
All mathematics in Section 7 matches the implementation exactly.*
