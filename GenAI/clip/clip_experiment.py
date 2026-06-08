# =============================================================================
# clip_experiment.py
# =============================================================================
# CLIP (Contrastive Language-Image Pre-Training) — From First Principles
# Reference implementation and experiment suite.
#
# Paper  : "Learning Transferable Visual Models From Natural Language
#            Supervision", Radford et al., OpenAI, 2021.
# ArXiv  : https://arxiv.org/abs/2103.00020
#
# PURPOSE
# -------
# This file teaches you every mathematical and algorithmic concept in CLIP by
# running four self-contained experiments.  All core experiments use only NumPy
# (no GPU, no model download) so you can run them anywhere in seconds.
# An optional fifth mode downloads the real openai/clip-vit-base-patch32 model
# from HuggingFace and runs identical experiments on actual learned weights.
#
# WHAT YOU WILL LEARN
# -------------------
#   Exp 1 — Contrastive Loss   : how CLIP trains from (image, text) pairs
#   Exp 2 — Zero-Shot          : classifying images with only text labels
#   Exp 3 — Temperature τ      : how sharpness/confidence is controlled
#   Exp 4 — Embedding Geometry : why "dog" is closer to "cat" than to "car"
#  (Exp 5)— Real CLIP          : same experiments on actual model weights
#
# USAGE
# -----
#   python clip_experiment.py            # NumPy simulation (no downloads)
#   python clip_experiment.py --real     # Real CLIP  (downloads ~600 MB)
#
# DEPENDENCIES
# ------------
#   Simulation only : numpy  (standard in every Python environment)
#   Real CLIP mode  : pip install torch transformers Pillow requests
#
# =============================================================================

import sys          # sys.argv — for reading the --real command-line flag
import math         # math module — imported for completeness; np handles math
import numpy as np  # NumPy — all linear algebra (vectors, matrices, norms)

# Fix the random seed so results are identical across runs.
# In real CLIP training the seed is NOT fixed — randomness is essential for
# sampling diverse batches, which drives the quality of contrastive negatives.
np.random.seed(42)


# =============================================================================
# SECTION 0 — MATHEMATICAL BUILDING BLOCKS
# =============================================================================
# Every function below implements a formula from the CLIP paper directly.
# Each one is small (≤5 lines), pure, and has no side-effects.
# We keep them separate from the experiments so they are easy to inspect,
# test, and reuse.
# =============================================================================

def l2_norm(v: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the unit hypersphere by dividing by its L2 norm.

    WHY THIS MATTERS IN CLIP
    ------------------------
    Both the image encoder and the text encoder end with an L2 normalisation
    step.  After normalisation every embedding vector has length exactly 1.0,
    i.e. it lives on the surface of a unit hypersphere in ℝ^512.

    Working on the unit sphere has three benefits:
      (a) Cosine similarity reduces to a simple dot product:
              cos(a, b) = (a·b) / (|a| |b|)  →  a·b   (when |a|=|b|=1)
          Dot products are extremely fast to compute in batched matrix form.
      (b) The model cannot cheat by making one embedding much larger in
          magnitude than others — every embedding gets exactly one "vote".
      (c) The temperature parameter τ has a well-defined, stable effect
          (see experiment_temperature for details).

    FORMULA
    -------
        v̂ = v / ‖v‖₂        where  ‖v‖₂ = sqrt(Σ vᵢ²)

    NUMERICAL NOTE
    --------------
    If v is the all-zeros vector, this divides by zero.  Real CLIP avoids
    this by using a small epsilon (1e-8) in the denominator, but we omit it
    here for clarity because our random vectors are never exactly zero.

    Args:
        v: Any real-valued numpy array of shape (d,).

    Returns:
        Unit vector with the same direction as v; shape (d,), ‖result‖₂ = 1.
    """
    return v / np.linalg.norm(v)   # np.linalg.norm defaults to the L2 (Frobenius) norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two L2-normalised vectors.

    WHY THIS MATTERS IN CLIP
    ------------------------
    CLIP measures how "aligned" an image embedding is with a text embedding
    using cosine similarity.  A value close to +1 means they describe the
    same concept; close to 0 means unrelated; negative values mean
    semantically opposite (rare in practice after training).

    FORMULA (general case)
    ----------------------
        cos(a, b) = (a · b) / (‖a‖₂ × ‖b‖₂)

    SIMPLIFICATION (when both vectors are L2-normalised)
    -----------------------------------------------------
    Since ‖a‖₂ = ‖b‖₂ = 1 after l2_norm, the denominator is 1 and:
        cos(a, b) = a · b   (just the dot product)

    This is why CLIP normalises all embeddings first — it turns the expensive
    cosine calculation into a cheap dot product, and allows the entire N×N
    similarity matrix to be computed as a single matrix multiplication:
        S = Ê_I @ Ê_T.T    (N×d  ×  d×N  →  N×N)

    RANGE
    -----
        -1  ≤  cos(a, b)  ≤  +1
    In practice, after CLIP training, image-text cosine similarities for
    correct pairs cluster around 0.20–0.35 on unit-normalised 512-d vectors.

    Args:
        a: L2-normalised vector, shape (d,).
        b: L2-normalised vector, shape (d,).

    Returns:
        Scalar float in [-1, +1].
    """
    return float(np.dot(a, b))   # np.dot on 1-D arrays is the inner product Σ aᵢbᵢ


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert a vector of raw scores (logits) into a probability distribution.

    WHY THIS MATTERS IN CLIP
    ------------------------
    CLIP uses softmax in two places:
      (a) During training: the cross-entropy loss applies softmax to each row
          and column of the N×N similarity logit matrix.
      (b) During zero-shot inference: softmax converts the K similarity scores
          (one per class label) into class probabilities P(class_k | image).

    FORMULA
    -------
        softmax(xᵢ; τ) = exp(xᵢ / τ) / Σⱼ exp(xⱼ / τ)

    TEMPERATURE SCALING
    -------------------
    Dividing by τ (tau) before the exponential controls the "sharpness":
      • τ → ∞  :  all probabilities equal 1/K  (maximum uncertainty)
      • τ → 0  :  probability mass concentrates on the highest-scoring class
                  (argmax behaviour)
      • τ = 1  :  standard softmax with no scaling
      • τ = 0.07:  CLIP's default — sharp but not a hard argmax

    NUMERICAL STABILITY TRICK
    -------------------------
    Computing exp(x) directly overflows for large x (e.g., exp(1000) = ∞).
    The standard fix: subtract the maximum value before exponentiating.
    This doesn't change the output because:
        exp(xᵢ - max) / Σⱼ exp(xⱼ - max)  =  exp(xᵢ) / Σⱼ exp(xⱼ)
    (the max cancels in numerator and denominator).

    Args:
        x:           1-D array of raw scores (logits), shape (K,).
        temperature: Positive scalar τ.  Default 1.0 = standard softmax.

    Returns:
        Probability array summing to 1.0, same shape as x.
    """
    x = np.array(x, dtype=float) / temperature   # scale logits by 1/τ
    x -= x.max()                                  # subtract max for numerical stability
    e = np.exp(x)                                 # element-wise exponential
    return e / e.sum()                            # normalise so probabilities sum to 1


def cross_entropy(logit_matrix: np.ndarray) -> float:
    """
    Compute the mean row-wise cross-entropy loss where the correct class for
    row i is always column i (the diagonal).

    WHY THIS MATTERS IN CLIP
    ------------------------
    This implements the image→text direction of the CLIP loss:
        "Given image i, which of the N texts is the correct match?"
    The correct text is always at position i (the diagonal of the matrix).

    The same function applied to logit_matrix.T gives the text→image loss:
        "Given text i, which of the N images is the correct match?"

    FORMULA (for one row i)
    -----------------------
        ℓᵢ = -log( exp(sᵢᵢ) / Σⱼ exp(sᵢⱼ) )
           = -sᵢᵢ + log( Σⱼ exp(sᵢⱼ) )

    This is equivalent to:  CrossEntropy( softmax(row_i),  one_hot(target=i) )

    MEAN OVER BATCH
    ---------------
    We average over all N rows to get a single scalar loss per direction.
    The full CLIP loss averages the image→text and text→image directions:
        L = (L_img + L_txt) / 2

    NUMERICAL STABILITY
    -------------------
    Same trick as softmax: subtract the row maximum before exp to avoid
    overflow.  The subtraction cancels in the log-softmax computation.

    Args:
        logit_matrix: N×N array of scaled similarities (already divided by τ).
                      logit_matrix[i, j] = cos(img_i, txt_j) / τ

    Returns:
        Scalar mean cross-entropy loss (lower = better alignment).
    """
    N = logit_matrix.shape[0]   # batch size = number of (image, text) pairs
    loss = 0.0

    for i in range(N):
        # Numerical stability: subtract row maximum before computing log-sum-exp
        row = logit_matrix[i] - logit_matrix[i].max()

        # -row[i]         → negative log-probability of the correct pair
        # log(exp(row).sum()) → log of the partition function (normalising constant)
        # Together: the negative log of the softmax probability at the diagonal
        loss += -row[i] + np.log(np.exp(row).sum())

    return loss / N   # average over the batch


def symmetric_clip_loss(sim_matrix: np.ndarray, tau: float) -> float:
    """
    The complete CLIP training objective (Section 2.4 of the CLIP paper).

    This is the single loss function that trains the entire model — both the
    image encoder and the text encoder — end-to-end using gradient descent.

    INTUITION
    ---------
    Given a batch of N (image, text) pairs, we want:
      • High similarity for the N matching pairs (the diagonal of sim_matrix)
      • Low similarity for the N² - N non-matching pairs (off-diagonal)

    We enforce this in BOTH directions simultaneously:
      • Image→text:  "Given image i, find its matching text among N options"
      • Text→image:  "Given text i, find its matching image among N options"

    WHY SYMMETRIC?
    --------------
    Using both directions means the model must satisfy both constraints at
    once, which makes the representations more robust.  Empirically, the
    symmetric loss gives 2–3% better downstream accuracy than one-directional.

    FORMULA
    -------
    Let S be the N×N cosine similarity matrix, sᵢⱼ = cos(img_i, txt_j).
    Scaled logit matrix:  L = S / τ

    Image→text loss:  L_img = (1/N) Σᵢ [ -Lᵢᵢ + log Σⱼ exp(Lᵢⱼ) ]
    Text→image loss:  L_txt = (1/N) Σⱼ [ -Lⱼⱼ + log Σᵢ exp(Lᵢⱼ) ]  (same, but on L.T)
    Total CLIP loss:  L = (L_img + L_txt) / 2

    SCALE EFFECT
    ------------
    With N=4096 pairs per batch (CLIP's actual training setting), each batch
    contains 4096 positive pairs and 4096²-4096 ≈ 16.7 million negatives.
    This vast sea of negatives is what makes the representations so
    discriminative — the model must push the correct match above ~16.7M
    distractors, every single training step.

    Args:
        sim_matrix: N×N numpy array.  sim_matrix[i,j] = cosine_sim(img_i, txt_j).
                    Values in [-1, +1] (before temperature scaling).
        tau:        Temperature scalar τ > 0.  CLIP initialises at 0.07 and
                    learns it during training.  It is equivalent to 1/learned_temp
                    in the CLIP paper's notation.

    Returns:
        Scalar symmetric CLIP loss.  Lower is better.
        A perfectly aligned batch would give loss ≈ 0.
        A random, untrained batch gives loss ≈ log(N) ≈ 3.69 for N=40.
    """
    logits  = sim_matrix / tau                     # scale similarities by 1/τ
    L_img   = cross_entropy(logits)                # rows are images,  columns are texts
    L_txt   = cross_entropy(logits.T)              # rows are texts,   columns are images
    return (L_img + L_txt) / 2.0                   # symmetric average


def print_section(title: str) -> None:
    """
    Print a visually distinct section header to stdout.

    Used throughout the experiments to separate output into readable blocks.
    No mathematical content — purely cosmetic.
    """
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


# =============================================================================
# SECTION 1 — HOW WE SIMULATE CLIP EMBEDDINGS WITHOUT A REAL MODEL
# =============================================================================
# All four simulation experiments create "fake" embeddings using Gaussian
# random vectors.  This is a valid proxy for several reasons:
#
#   1. High-dimensional random vectors are approximately orthogonal.
#      In ℝ^512, two random unit vectors have expected cosine similarity ≈ 0.
#      This simulates the vast "empty space" in real embedding spaces.
#
#   2. We model semantic relatedness by making related embeddings share a
#      component with a common "concept anchor" vector.  For example:
#          dog_image  ≈  dog_concept + small_noise
#          dog_text   ≈  dog_concept + different_small_noise
#      Both are close to dog_concept, so they're close to each other — exactly
#      what a well-trained CLIP model produces.
#
#   3. The mathematics of softmax, cross-entropy, and cosine similarity are
#      identical regardless of whether the vectors came from a neural network
#      or a random generator.
#
# This means experiments 1–4 faithfully demonstrate the CLIP loss mechanics
# even without any model weights.
# =============================================================================


# =============================================================================
# EXPERIMENT 1 — CONTRASTIVE LOSS ON A TRAINING BATCH
# =============================================================================

def experiment_contrastive_loss(dim: int = 512, N: int = 4, tau: float = 0.07) -> None:
    """
    Simulate one CLIP training step on a mini-batch of N image-text pairs.

    This experiment demonstrates:
      (a) What the N×N cosine similarity matrix looks like for well-aligned
          vs. randomly initialised embeddings.
      (b) How the symmetric cross-entropy loss numerically penalises
          off-diagonal similarities and rewards diagonal similarities.
      (c) The ratio between trained and untrained losses — showing how large
          the gradient signal is at the start of training.

    WHAT CLIP ACTUALLY DOES AT TRAINING TIME
    ----------------------------------------
    At each training step, CLIP:
      1. Samples a batch of N (image, text) pairs from its 400M-pair dataset.
      2. Passes all N images through the image encoder → N unit vectors.
      3. Passes all N texts through the text encoder → N unit vectors.
      4. Computes the N×N cosine similarity matrix S.
      5. Scales S by 1/τ to get logits L.
      6. Computes the symmetric cross-entropy loss on L.
      7. Backpropagates gradients through both encoders and updates weights.

    Steps 1–6 are exactly what this function simulates.  Step 7 requires
    automatic differentiation (PyTorch/JAX) and the actual model parameters.

    SIMULATION DESIGN
    -----------------
    We create N "concept" vectors — one per semantic category — then perturb
    each one twice (once for the image encoder output, once for the text
    encoder output) with small Gaussian noise.

        concept_k  ~  Uniform(unit sphere in ℝ^512)
        img_k      =  l2_norm( concept_k + noise * ε₁ )   ε₁ ~ N(0, I)
        txt_k      =  l2_norm( concept_k + noise * ε₂ )   ε₂ ~ N(0, I)

    Because noise=0.05 is small, img_k and txt_k are both very close to
    concept_k, hence very close to each other — they form a correct pair.
    All cross-pairs (img_i, txt_j for i≠j) are close to different concepts,
    hence far apart on the sphere.

    Args:
        dim:  Dimensionality of the embedding space.  CLIP uses 512.
        N:    Batch size.  Real CLIP trains with N=4096 (or 32768 for ViT-L).
              We use N=4 so the matrix is readable in a terminal.
        tau:  Temperature.  CLIP's learned temperature converges to ~0.07.
    """
    print_section("EXPERIMENT 1: Contrastive Loss on a Batch of 4 Pairs")

    # ── Step 1: Create semantic concept anchors ────────────────────────────────
    # Each concept is a unit vector drawn from a standard Gaussian, then
    # normalised.  In 512 dimensions, any two random unit vectors have
    # expected dot product ≈ 0, so concepts are approximately orthogonal.
    concepts = [l2_norm(np.random.randn(dim)) for _ in range(N)]

    # noise controls how tightly image/text embeddings cluster around the concept.
    # noise=0.05 means the perturbation has magnitude ~0.05 before normalisation,
    # so the resulting embedding is very close to the original concept vector.
    # In real CLIP, the encoders learn to produce embeddings that are similarly
    # close to a shared semantic point.
    noise = 0.05

    # ── Step 2: Create image and text embeddings ───────────────────────────────
    # Independent noise for images vs. texts — same concept, different path.
    # This mirrors reality: the ViT and the GPT transformer are separate networks
    # with separate parameters, but both converge to the same semantic point.
    imgs = [l2_norm(c + noise * np.random.randn(dim)) for c in concepts]
    txts = [l2_norm(c + noise * np.random.randn(dim)) for c in concepts]

    # Human-readable labels for the terminal output
    labels = ["dog photo", "car photo", "mountain", "pizza"]

    # ── Step 3: Build the N×N cosine similarity matrix ────────────────────────
    # S[i, j]  =  cosine_sim(image_i, text_j)
    # This is the raw similarity BEFORE temperature scaling.
    # In real CLIP this is computed as a single batched matrix multiply:
    #     S = imgs_matrix @ txts_matrix.T     (both matrices are N × dim)
    # We use a Python loop here for clarity.
    S = np.array([
        [cosine_sim(imgs[i], txts[j]) for j in range(N)]
        for i in range(N)
    ])

    # ── Step 4: Display the scaled logit matrix ────────────────────────────────
    # We display S/τ (the logits, not the raw similarities) because the loss
    # operates on the logits.  The diagonal entries (✓) should be the largest
    # in each row and column — that's exactly what the loss pushes toward.
    print(f"\nCosine similarity matrix (τ={tau}, scaled = S/τ):")
    print(f"  {'':12}", end="")
    for lbl in labels:
        print(f"{lbl:>12}", end="")
    print()
    for i in range(N):
        print(f"  {labels[i]:12}", end="")
        for j in range(N):
            mark = " ✓" if i == j else "  "   # ✓ = the correct pair
            print(f"{S[i,j]/tau:10.2f}{mark}", end="")
        print()

    # ── Step 5: Compute the symmetric CLIP loss ────────────────────────────────
    loss = symmetric_clip_loss(S, tau)
    print(f"\nSymmetric CLIP loss (well-aligned batch): {loss:.5f}")
    print("  → Lower is better.  A perfect batch with identical pairs gives loss → 0.")

    # ── Step 6: Compare to a random (untrained) batch ─────────────────────────
    # Before any training, the encoders output essentially random unit vectors.
    # We simulate this by generating entirely independent random vectors (no
    # shared concept anchor).  The loss should be much higher.
    bad_imgs = [l2_norm(np.random.randn(dim)) for _ in range(N)]
    bad_txts = [l2_norm(np.random.randn(dim)) for _ in range(N)]
    S_bad    = np.array([
        [cosine_sim(bad_imgs[i], bad_txts[j]) for j in range(N)]
        for i in range(N)
    ])
    bad_loss = symmetric_clip_loss(S_bad, tau)

    print(f"Symmetric CLIP loss (random/untrained batch):  {bad_loss:.5f}")
    print(f"Ratio (untrained / trained):  {bad_loss/loss:.1f}×")
    print("  → The contrastive loss provides a very large gradient signal early in training.")
    print("  → As training progresses, the ratio shrinks toward 1.0 (loss converges).")


# =============================================================================
# EXPERIMENT 2 — ZERO-SHOT CLASSIFICATION
# =============================================================================

def experiment_zero_shot(dim: int = 512, tau: float = 0.07) -> None:
    """
    Demonstrate zero-shot image classification — CLIP's signature capability.

    WHAT "ZERO-SHOT" MEANS
    ----------------------
    Traditional classifiers require labelled training examples for every class.
    "Zero-shot" means we can classify into NEW categories at inference time
    without any training on those categories.  We only need the category names
    written as natural language.

    HOW IT WORKS
    ------------
    Given a query image and K candidate class names:
      1. Encode the image:  ê_I = ImageEncoder(image)  ∈ ℝ^512, unit norm
      2. For each class k, form the prompt "a photo of a {class_k}" and encode:
              ê_T^(k) = TextEncoder("a photo of a " + class_k)
      3. Compute K cosine similarities:  sₖ = ê_I · ê_T^(k)
      4. Apply softmax with temperature τ:  P(class_k | image) = softmax(s/τ)[k]
      5. Predict: argmax_k P(class_k | image)

    Note: no model parameters are updated.  We are just doing K nearest-neighbour
    lookup in the shared embedding space, where "distance" is cosine similarity.

    THE PROMPT "a photo of a {class}"
    ----------------------------------
    Why not just use the class name directly?
    Because CLIP was trained on image CAPTIONS, not on bare class names.
    Captions start with phrases like "a photo of", "an image showing", etc.
    Using the same phrasing at inference time reduces the train/test distribution
    gap and typically improves accuracy by 3–5 percentage points on ImageNet.
    This is called "prompt engineering" for CLIP.

    SIMULATION DESIGN
    -----------------
    We build a small "concept atlas" of 6 semantic concepts.  The query image
    is a noisy version of concept[0] ("dog").  The text embeddings for each
    class are also noisy versions of their respective concept.

    For the "cat" class, we deliberately add a component toward the "dog"
    concept (0.15 × concepts[0]) to simulate the fact that cats and dogs are
    semantically related — they should be closer than dogs and cars.

    Args:
        dim:  Embedding dimensionality.  CLIP uses 512.
        tau:  Temperature.  Same value as training.
    """
    print_section("EXPERIMENT 2: Zero-Shot Classification")

    # ── Step 1: Build the concept atlas ───────────────────────────────────────
    # Each concept is an independent random unit vector.  In 512 dimensions,
    # any two are approximately orthogonal, so concepts don't interfere.
    concept_names = ["dog", "car", "mountain", "pizza", "cat", "airplane"]
    concepts      = [l2_norm(np.random.randn(dim)) for _ in concept_names]

    # ── Step 2: Simulate the query image (a dog photo) ────────────────────────
    # The image encoder output is close to concepts[0] (dog) with a tiny
    # noise perturbation.  In reality, the ViT processes pixel patches and
    # produces a similar cluster-near-concept structure after training.
    query_image = l2_norm(concepts[0] + 0.04 * np.random.randn(dim))

    # ── Step 3: Build text embeddings for each class ──────────────────────────
    # Each text embedding clusters near its concept.  We use slightly less noise
    # for "dog" (0.06) and more for the others (0.10) to model the fact that
    # the correct class text is the closest match.
    class_texts = {
        name: l2_norm(concepts[i] + (0.06 if i == 0 else 0.10) * np.random.randn(dim))
        for i, name in enumerate(concept_names)
    }

    # Special case: "cat" gets a component toward "dog" to model semantic proximity.
    # cats and dogs are both domestic animals; real CLIP embeddings reflect this.
    class_texts["cat"] = l2_norm(
        concepts[4]           +   # cat's own concept
        0.15 * concepts[0]    +   # some overlap with the dog concept
        0.10 * np.random.randn(dim)
    )

    # ── Step 4: Compute cosine similarities ───────────────────────────────────
    # For each class label k:  sₖ = ê_I · ê_T^(k)
    # This is identical to what CLIP does at inference time.
    sims = {name: cosine_sim(query_image, vec) for name, vec in class_texts.items()}

    # ── Step 5: Softmax → class probabilities ─────────────────────────────────
    # We softmax over the K=6 similarity scores to get a proper probability
    # distribution.  Temperature τ controls sharpness (see Experiment 3).
    probs = softmax(list(sims.values()), temperature=tau)

    # ── Step 6: Display results ────────────────────────────────────────────────
    # Sort by probability descending so the highest-confidence class is first.
    entries = sorted(zip(sims.keys(), sims.values(), probs), key=lambda x: -x[2])

    print("\nQuery: dog image embedding (simulated)")
    print(f'Prompt template applied: "a photo of a {{class}}"')
    print()
    print(f"  {'Class':10s}  {'Similarity':>10}  {'P(class|image)':>14}  Bar")
    print(f"  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*40}")
    for name, sim, prob in entries:
        bar = "█" * int(prob * 50)     # visual bar width proportional to probability
        print(f"  {name:10s}  {sim:+10.4f}  {prob:14.4f}  {bar}")

    winner = entries[0][0]
    correct = "✓  (correct!)" if winner == "dog" else "✗  (wrong)"
    print(f"\n→ Predicted class: \"{winner}\"  {correct}")
    print(f"  Confidence in 'dog': {entries[0][2]*100:.1f}%")


# =============================================================================
# EXPERIMENT 3 — TEMPERATURE τ ABLATION
# =============================================================================

def experiment_temperature(dim: int = 512) -> None:
    """
    Systematically vary the temperature τ and observe its effect on the output
    probability distribution.

    WHAT IS THE TEMPERATURE PARAMETER?
    ------------------------------------
    Temperature τ appears in the CLIP loss as:
        logits = similarity_matrix / τ

    It is a LEARNABLE scalar parameter — CLIP initialises it at log(1/0.07) ≈ 2.65
    in log-space and optimises it alongside the encoder weights.

    INTUITION: τ AS A THERMOSTAT
    -----------------------------
    Think of the probability distribution over classes as a physical system:
      • High temperature (τ ≈ 2.0): the system is "hot" — probabilities spread
        nearly uniformly over all classes.  The model is uncertain.
      • Low temperature  (τ ≈ 0.01): the system is "cold" — nearly all
        probability mass concentrates on the single highest-scoring class.
        Effectively an argmax operation.
      • Goldilocks zone  (τ ≈ 0.07): sharp enough to make confident predictions,
        but soft enough to preserve gradient signal through the loss function.

    WHY τ MUST BE LEARNED (NOT FIXED)
    -----------------------------------
    The appropriate temperature depends on:
      1. The actual cosine similarity values, which change as the encoders train.
      2. The number of classes K at inference time — more classes need sharper
         distributions to avoid probability mass dilution.
    CLIP learns τ jointly with the encoder weights so it auto-adjusts.

    ENTROPY AS A MEASURE OF UNCERTAINTY
    ------------------------------------
    We report Shannon entropy:  H(p) = -Σₖ pₖ log(pₖ)
      • H = 0     : perfectly certain (all mass on one class)
      • H = log(K): maximum uncertainty (uniform distribution over K classes)
    For K=6 classes, maximum entropy = log(6) ≈ 1.79 nats.

    Args:
        dim: Embedding dimensionality.  Results qualitatively identical for any dim.
    """
    print_section("EXPERIMENT 3: Temperature τ — Sharpness vs. Calibration")

    # Create a query image and 6 class text embeddings.
    # We reuse the same similarities for all temperature values so the ONLY
    # thing changing is τ — a clean ablation.
    concepts = [l2_norm(np.random.randn(dim)) for _ in range(6)]
    query    = l2_norm(concepts[0] + 0.04 * np.random.randn(dim))
    texts    = [l2_norm(c + 0.08 * np.random.randn(dim)) for c in concepts]

    # Raw cosine similarities — these are FIXED for all τ values below
    sims = np.array([cosine_sim(query, t) for t in texts])

    print(f"\nRaw cosine similarities (fixed, τ-independent):")
    print("  [" + ", ".join(f"{s:+.4f}" for s in sims) + "]")
    print("  → Class 0 ('dog') has the highest similarity: it's the correct class.")
    print()
    print(f"  {'τ':>6}  {'P_top':>7}  {'H (entropy)':>11}  {'Interpretation'}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*11}  {'-'*36}")

    for tau in [2.00, 0.50, 0.10, 0.07, 0.03, 0.01]:
        probs   = softmax(sims, temperature=tau)          # apply temperature scaling
        p_top   = float(probs.max())                      # probability of the top class
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))  # Shannon entropy in nats

        # Classify the regime for intuition
        if   tau >= 1.0:  note = "too uncertain — near-uniform distribution"
        elif tau >= 0.3:  note = "moderate — losing discrimination"
        elif tau >= 0.07: note = "CLIP regime — sharp and well-calibrated"
        elif tau >= 0.02: note = "overconfident — gradient may vanish"
        else:             note = "≈ argmax — no gradient signal"

        print(f"  {tau:>6.2f}  {p_top:>7.4f}  {entropy:>11.4f}  {note}")

    print()
    print("Key takeaway: CLIP's τ ≈ 0.07 balances:")
    print("  1. Sharp enough predictions to correctly rank the true match above ~16M negatives")
    print("  2. Soft enough distribution to maintain non-zero gradients for learning")


# =============================================================================
# EXPERIMENT 4 — GEOMETRY OF THE SHARED EMBEDDING SPACE
# =============================================================================

def experiment_embedding_geometry(dim: int = 512) -> None:
    """
    Verify that CLIP's shared embedding space preserves semantic structure.

    THE CORE CLAIM
    --------------
    After training, the shared embedding space is not a random jumble — it
    has SEMANTIC GEOMETRY:
      • Concepts within the same category (e.g., dog, cat, bird) cluster
        together in one region of the sphere.
      • Concepts from different categories (e.g., dog vs. car) are far apart.
      • The geometry is SHARED across modalities — the dog IMAGE cluster and
        the dog TEXT cluster overlap, not just parallel.

    This is what makes zero-shot classification work: a dog image lands near
    all dog-related text, not just the exact caption it was trained with.

    SIMULATION DESIGN
    -----------------
    We model three semantic categories:
      • Animals  (dog, cat, bird)  → built around `animal_anchor` vector
      • Vehicles (car, truck, airplane) → built around `vehicle_anchor` vector
      • Food     (pizza, pasta)    → built around `food_anchor` vector

    Each concept in a category = category_anchor + small_noise.
    So within-category concepts have high cosine similarity (share the anchor).
    Cross-category concepts have low cosine similarity (their anchors are random
    and approximately orthogonal in high-dimensional space).

    EXPECTED RESULT
    ---------------
    Query = dog image embedding (≈ animal_anchor + noise)

    Cosine similarities should rank:
        dog (same exact concept)         ≈ 0.7–0.9   [highest]
        cat, bird (same category)        ≈ 0.1–0.2   [medium]
        pasta, pizza (different category) ≈ 0.0–0.05 [low]
        car, truck, airplane             ≈ 0.0–0.05  [low]

    This mirrors what real CLIP shows in t-SNE visualisations of its
    embedding space.

    Args:
        dim: Embedding dimensionality.
    """
    print_section("EXPERIMENT 4: Geometry of the Shared Embedding Space")

    # ── Step 1: Build category anchor vectors ─────────────────────────────────
    # Three random unit vectors, one per semantic category.
    # In ℝ^512, these are approximately orthogonal (dot product ≈ 0).
    # They represent the "centre of mass" of each semantic cluster.
    animal_anchor  = l2_norm(np.random.randn(dim))
    vehicle_anchor = l2_norm(np.random.randn(dim))
    food_anchor    = l2_norm(np.random.randn(dim))

    # ── Step 2: Build individual concept embeddings ────────────────────────────
    # Each concept = its category's anchor + small independent Gaussian noise.
    # noise=0.10 means the concept is ~5–10° away from the anchor on the sphere.
    # Within-category pairs therefore have cosine similarity ≈ cos(10°) ≈ 0.98
    # when unnormalised, but after normalisation the actual values depend on dim.
    concepts = {
        # Animal cluster — all share animal_anchor
        "dog":      l2_norm(animal_anchor  + 0.10 * np.random.randn(dim)),
        "cat":      l2_norm(animal_anchor  + 0.10 * np.random.randn(dim)),
        "bird":     l2_norm(animal_anchor  + 0.10 * np.random.randn(dim)),
        # Vehicle cluster — all share vehicle_anchor
        "car":      l2_norm(vehicle_anchor + 0.10 * np.random.randn(dim)),
        "truck":    l2_norm(vehicle_anchor + 0.10 * np.random.randn(dim)),
        "airplane": l2_norm(vehicle_anchor + 0.10 * np.random.randn(dim)),
        # Food cluster — all share food_anchor
        "pizza":    l2_norm(food_anchor    + 0.10 * np.random.randn(dim)),
        "pasta":    l2_norm(food_anchor    + 0.10 * np.random.randn(dim)),
    }

    # ── Step 3: Simulate a query dog image embedding ───────────────────────────
    # The image is very close to the "dog" concept but with small noise
    # (simulating that the ViT encoder is not a perfect lookup table).
    query_vec = l2_norm(concepts["dog"] + 0.04 * np.random.randn(dim))

    # ── Step 4: Compute cosine similarities to all concepts ───────────────────
    scores = {name: cosine_sim(query_vec, vec) for name, vec in concepts.items()}

    # ── Step 5: Display results sorted by similarity ──────────────────────────
    print(f"\nQuery: 'dog' image embedding  (looking at all concept text embeddings)")
    print()
    print(f"  {'Concept':10s}  {'Category':8s}  {'cos sim':>8}  Bar")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*60}")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        # Map each concept back to its category for display
        category = (
            "animal"  if name in ("dog", "cat", "bird")
            else "vehicle" if name in ("car", "truck", "airplane")
            else "food"
        )
        # Bar length: linearly map similarities in [-0.1, +0.8] to bar width
        bar = "█" * max(1, int((score + 0.1) * 100))
        print(f"  {name:10s}  {category:8s}  {score:+8.4f}  {bar}")

    # ── Step 6: Compute category-level aggregates ──────────────────────────────
    print()
    print("Category-level average cosine similarities:")
    animal_avg  = float(np.mean([scores[n] for n in ("cat", "bird")]))
    vehicle_avg = float(np.mean([scores[n] for n in ("car", "truck", "airplane")]))
    food_avg    = float(np.mean([scores[n] for n in ("pizza", "pasta")]))

    print(f"  dog image → other animals  : {animal_avg:+.4f}   ← same category, medium sim")
    print(f"  dog image → vehicles       : {vehicle_avg:+.4f}   ← different category, low sim")
    print(f"  dog image → food           : {food_avg:+.4f}   ← different category, low sim")
    print()
    print("This hierarchical structure is exactly what t-SNE plots of real")
    print("CLIP embeddings show: tight clusters within categories, separated")
    print("clusters between categories, in the SAME space for images and text.")


# =============================================================================
# EXPERIMENT 5 — REAL CLIP (OPTIONAL, REQUIRES MODEL DOWNLOAD)
# =============================================================================

def run_real_clip() -> None:
    """
    Run all experiments on the actual openai/clip-vit-base-patch32 weights.

    This requires downloading ~600 MB of model weights from HuggingFace the
    first time; subsequent runs use the local cache (~/.cache/huggingface/).

    WHAT MODEL IS USED?
    -------------------
    openai/clip-vit-base-patch32
      • Image encoder  : ViT-B/32  (Vision Transformer, base size, 32px patches)
        - Input : 224×224 RGB image
        - Patches: 7×7 grid = 49 patches of 32×32 pixels each
        - Hidden dim: 768
        - Layers: 12 transformer layers
        - Output: 512-d embedding (via linear projection of [CLS] token)
      • Text encoder   : GPT-style transformer (12 layers, 512-d)
        - Input : up to 77 BPE tokens from a vocab of 49,152
        - Output: 512-d embedding (via linear projection of [EOT] token)
      • Total parameters: ~150M
      • Training data   : WIT (WebImageText) — 400M (image, text) pairs
                          scraped from the internet

    The larger ViT-L/14 model (used for CLIP's best ImageNet results) is ~800M
    parameters and requires ~3GB download.

    WHAT THIS FUNCTION RUNS
    -----------------------
      (a) Zero-shot classification on a real dog photo (downloaded from Wikipedia)
          with 7 class labels.
      (b) Prompt engineering ablation: compare different prompt templates for
          the same class and see how P(dog|image) changes.
      (c) Text-text similarity: show that the TEXT encoder alone produces
          meaningful semantic distances, even without any image input.
          e.g., cos("dog", "golden retriever") >> cos("dog", "automobile")

    DEPENDENCIES
    ------------
        pip install torch transformers Pillow requests
    """
    print_section("EXPERIMENT 5 (REAL CLIP): openai/clip-vit-base-patch32")

    # ── Import optional dependencies ───────────────────────────────────────────
    # We import here (not at module level) so the simulation experiments still
    # run even without torch/transformers installed.
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        import requests
        from io import BytesIO
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with:  pip install torch transformers Pillow requests")
        return

    # ── Load model and processor ───────────────────────────────────────────────
    # CLIPModel      : contains both the image encoder (ViT) and text encoder (GPT)
    # CLIPProcessor  : handles pre-processing:
    #                    - images → resize, centre-crop, normalise pixel values
    #                    - text   → BPE tokenise, pad to 77 tokens
    # Both are downloaded once and cached in ~/.cache/huggingface/
    print("Loading model weights (~600 MB on first run, then cached)...")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()   # disable dropout — we are doing inference, not training
    print("Model loaded successfully.\n")

    # ── Download a sample image ────────────────────────────────────────────────
    # We use a public-domain dog photo from Wikimedia Commons.
    # The URL is a 320px-wide thumbnail to keep the download fast.
    IMAGE_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
    )
    print(f"Downloading test image: {IMAGE_URL}")
    try:
        response  = requests.get(IMAGE_URL, timeout=15)
        response.raise_for_status()                         # raise on HTTP errors
        image     = Image.open(BytesIO(response.content)).convert("RGB")
        # CLIPProcessor will resize/crop this to 224×224 internally.
        print(f"Image downloaded: {image.size[0]}×{image.size[1]} pixels\n")
        has_image = True
    except Exception as e:
        print(f"Could not download image: {e}")
        print("Skipping image experiments — text-only experiments will still run.\n")
        has_image = False

    # ── Sub-experiment A: Zero-shot classification ─────────────────────────────
    if has_image:
        print("─" * 56)
        print("A) Zero-shot classification")
        print("─" * 56)

        class_labels = ["dog", "cat", "car", "mountain", "pizza", "bird", "boat"]

        # CLIP performs best with full sentence prompts, not bare class names.
        # The phrase "a photo of a" closely matches the captions in its training data.
        prompts = [f"a photo of a {c}" for c in class_labels]

        # processor() simultaneously:
        #   - Decodes and resizes the image to 224×224, normalises pixels
        #   - Tokenises each text with BPE, pads/truncates to 77 tokens
        # return_tensors="pt" returns PyTorch tensors (required by CLIPModel)
        inputs = processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,          # pad text sequences to the same length in batch
        )

        # torch.no_grad() disables gradient computation — we don't need it
        # for inference and it saves memory and compute.
        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.logits_per_image is the scaled similarity matrix (S/τ) with
        # shape [num_images, num_texts] = [1, 7] in this case.
        # .softmax(dim=1) converts to probabilities over the 7 text prompts.
        # .squeeze() removes the batch dimension → shape [7]
        probs   = outputs.logits_per_image.softmax(dim=1).squeeze().numpy()
        entries = sorted(zip(class_labels, probs), key=lambda x: -x[1])

        print(f"\nInput image: dog photo from Wikipedia")
        print(f"Candidates:  {class_labels}")
        print()
        for name, prob in entries:
            bar = "█" * int(float(prob) * 50)
            print(f"  {name:10s}  P={float(prob):.4f}  {bar}")
        print(f"\n→ Prediction: \"{entries[0][0]}\"")

    # ── Sub-experiment B: Prompt engineering ──────────────────────────────────
    if has_image:
        print()
        print("─" * 56)
        print("B) Prompt engineering ablation")
        print("─" * 56)
        print("  Same image, same competitor classes (cat, car), different prompts.")
        print("  Observe how P(dog|image) changes with prompt specificity.\n")

        prompt_variants = [
            "dog",                                     # bare class name (suboptimal)
            "a dog",                                   # slightly better — article added
            "a photo of a dog",                        # CLIP's standard template
            "a high quality photo of a dog",           # more specific
            "a golden retriever dog playing outdoors", # very specific, overfitted
        ]

        for prompt in prompt_variants:
            inputs = processor(
                text=[prompt, "a cat", "a car"],   # dog prompt vs two distractors
                images=image,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                outputs = model(**inputs)

            probs    = outputs.logits_per_image.softmax(dim=1).squeeze().numpy()
            dog_prob = float(probs[0])   # position 0 = the dog prompt
            bar      = "█" * int(dog_prob * 40)
            print(f"  P(dog)={dog_prob:.3f}  {bar}  | prompt: \"{prompt}\"")

        print()
        print("Key insight: 'a photo of a dog' reliably outperforms the bare word 'dog'")
        print("because CLIP's training data consisted of natural image captions,")
        print("not bare nouns.  This is prompt engineering for zero-shot models.")

    # ── Sub-experiment C: Text-text similarity ─────────────────────────────────
    print()
    print("─" * 56)
    print("C) Text-text cosine similarity in the shared embedding space")
    print("─" * 56)
    print("  (No image needed — shows that the text encoder alone captures")
    print("   semantic structure, usable for text retrieval tasks.)\n")

    # Pairs to compare — we expect high sim for related concepts, low for unrelated.
    pairs = [
        ("dog",              "golden retriever"),     # specific vs. general, same animal
        ("dog",              "cat"),                  # different animals, same category
        ("dog",              "automobile"),           # animal vs. vehicle
        ("car",              "truck"),                # both vehicles
        ("car",              "apple"),                # unrelated
        ("machine learning", "neural network"),       # both ML concepts
        ("machine learning", "cooking recipe"),       # ML vs. unrelated domain
    ]

    # Deduplicate all unique texts, encode them in one batch (efficient).
    # We use a set comprehension and convert to list for stable ordering.
    unique_texts = list(dict.fromkeys(t for pair in pairs for t in pair))

    # get_text_features() runs only the TEXT encoder (no image encoder).
    # Output shape: [num_texts, embedding_dim] = [N, 512] in this case.
    inputs = processor(text=unique_texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)    # shape: [N, 512]
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalise → unit sphere

    # Build a text → index mapping for convenient lookup
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}

    print(f"  {'Text A':20s}  {'Text B':20s}  {'cos sim':>8}  Interpretation")
    print(f"  {'-'*20}  {'-'*20}  {'-'*8}  {'-'*25}")
    for a, b in pairs:
        # Cosine similarity = dot product of unit-normalised vectors
        sim = float(feats[text_to_idx[a]] @ feats[text_to_idx[b]])

        # Qualitative interpretation
        if   sim > 0.80: note = "very high — near-synonyms"
        elif sim > 0.60: note = "high — same category"
        elif sim > 0.30: note = "moderate — loosely related"
        elif sim > 0.10: note = "low — different domains"
        else:            note = "very low — unrelated"

        print(f"  {a:20s}  {b:20s}  {sim:+8.4f}  {note}")

    print()
    print("This text-only embedding is why CLIP can also do text retrieval,")
    print("cross-lingual matching (with multilingual models), and semantic")
    print("text search — the same embedding space that handles images also")
    print("organises language semantically.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║          CLIP From First Principles — Experiment Suite       ║
║          github.com / research reference implementation      ║
╚══════════════════════════════════════════════════════════════╝

Experiments 1–4 run immediately (NumPy only, no downloads needed).
Pass --real to also run Experiment 5 with actual model weights.
""")

    # ── Simulation experiments (always run) ───────────────────────────────────
    # These four experiments use only NumPy and execute in < 1 second.
    # They faithfully demonstrate the CLIP mathematics regardless of whether
    # any model weights are available.
    experiment_contrastive_loss()   # Exp 1: the training loss
    experiment_zero_shot()          # Exp 2: zero-shot classification procedure
    experiment_temperature()        # Exp 3: effect of temperature τ
    experiment_embedding_geometry() # Exp 4: semantic structure of embedding space

    # ── Real CLIP (optional) ──────────────────────────────────────────────────
    if "--real" in sys.argv:
        # Run Experiment 5: download real model weights and run identical experiments
        # on actual CLIP embeddings from openai/clip-vit-base-patch32.
        run_real_clip()
    else:
        # Guide the user on how to run the real model mode
        print()
        print("─" * 64)
        print("To run Experiment 5 with the real CLIP model:")
        print()
        print("  pip install torch transformers Pillow requests")
        print("  python clip_experiment.py --real")
        print()
        print("First run downloads ~600 MB; subsequent runs use cached weights.")
        print("GPU not required — CPU inference works fine for these experiments.")
        print("─" * 64)
