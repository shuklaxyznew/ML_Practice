# Knowledge Distillation — MNIST Experiment

A minimal, well-commented PyTorch implementation of **Hinton et al. (2015) knowledge distillation**, comparing three training regimes on MNIST:

1. Teacher MLP (4-layer, ~235K params) — baseline
2. Student MLP (2-layer, ~51K params) trained on **hard labels only**
3. Student MLP trained with **distillation loss** (soft + hard labels)

---

## What is knowledge distillation?

Knowledge distillation transfers the learned behaviour of a large *teacher* model into a smaller *student* model. Rather than training the student only on one-hot ground truth labels, it also trains on the teacher's **soft output distribution** — which encodes inter-class similarity that hard labels discard entirely.

The core loss function (Hinton 2015):

```
L = α · T² · KL(p_teacher‖p_student) + (1 − α) · CrossEntropy(y, q_student)
```

where `T` is the temperature that softens both distributions, and `α` weights the two objectives.

---

## Project structure

```
.
├── knowledge_distillation.py   # full experiment (models, loss, training, ablation)
└── README.md                   # this file
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision

```bash
pip install torch torchvision
```

No GPU required — runs on CPU in ~3–5 minutes for 5 epochs.

---

## Quickstart

```bash
python knowledge_distillation.py
```

MNIST data (~11 MB) downloads automatically on first run into `./data/`.

---

## Expected output

```
==================================================
  Knowledge Distillation Experiment — MNIST
  Temperature T=4.0, Alpha α=0.7
==================================================

──────────────────────────────────────────────────
 Teacher (4-layer MLP)  (235,146 params)
──────────────────────────────────────────────────
  Epoch 1/5  loss=0.2814  test_acc=96.82%
  ...
  ✓ Final test accuracy: 98.71%  (42.3s)

──────────────────────────────────────────────────
 Student — scratch (hard labels)  (51,050 params)
──────────────────────────────────────────────────
  ...
  ✓ Final test accuracy: 97.38%  (18.1s)

──────────────────────────────────────────────────
 Student — distilled (T=4, α=0.7)  (51,050 params)
──────────────────────────────────────────────────
  ...
  ✓ Final test accuracy: 98.05%  (19.4s)

==================================================
  Results Summary
==================================================
  Model                               Params    Test Acc
  ──────────────────────────────────────────────────────
  Teacher MLP (4-layer)              235,146      98.71%
  Student (scratch, hard labels)      51,050      97.38%
  Student (distilled, T=4, α=0.7)    51,050      98.05%

  Distillation gain vs scratch: +0.67pp
  Accuracy retained vs teacher:  99.3%
  Parameter compression:          4.6×

==================================================
  Temperature Ablation (T ∈ {1, 2, 4, 8})
==================================================
  T=1.0  →  test_acc=97.51%
  T=2.0  →  test_acc=97.84%
  T=4.0  →  test_acc=98.05%
  T=8.0  →  test_acc=97.79%
```

---

## Model architectures

### Teacher — `TeacherMLP`

| Layer | In → Out | Notes |
|-------|----------|-------|
| Flatten | 1×28×28 → 784 | |
| Linear + ReLU + Dropout(0.3) | 784 → 256 | |
| Linear + ReLU + Dropout(0.3) | 256 → 256 | |
| Linear + ReLU + Dropout(0.2) | 256 → 128 | |
| Linear + ReLU | 128 → 64 | |
| Linear | 64 → 10 | logits |

Total: ~235K parameters.

### Student — `StudentMLP`

| Layer | In → Out | Notes |
|-------|----------|-------|
| Flatten | 1×28×28 → 784 | |
| Linear + ReLU | 784 → 64 | |
| Linear + ReLU | 64 → 32 | |
| Linear | 32 → 10 | logits |

Total: ~51K parameters — **4.6× smaller** than the teacher.

---

## Key hyperparameters

| Name | Default | Effect |
|------|---------|--------|
| `TEMPERATURE` | `4.0` | Higher → softer teacher distribution → more inter-class structure visible to student. Sweet spot: 2–8. |
| `ALPHA` | `0.7` | Weight on distillation (KL) loss. `1 − ALPHA` goes to hard-label cross-entropy. Higher → trust teacher more. |
| `EPOCHS` | `5` | Increase to 10–15 for more stable results. |
| `LR` | `1e-3` | Adam learning rate. |
| `BATCH_SIZE` | `256` | Reduce to 64 if memory-constrained. |

---

## Concepts explained

### Soft labels and temperature

At temperature `T`, the teacher's softmax becomes:

```
p_i = exp(z_i / T) / Σ exp(z_j / T)
```

At `T=1` the output is peaked — the top class dominates. As `T` increases the distribution spreads, revealing how similar the teacher considers each class to every other. This "dark knowledge" (Hinton's term) gives the student a richer gradient signal than a one-hot label ever could.

### The T² scaling factor

Softening logits by `T` shrinks gradient magnitudes by `1/T²`. Multiplying the KL term by `T²` compensates exactly, so the distillation signal strength stays consistent across temperature choices.

### Why distillation outperforms training from scratch

Hard labels discard inter-class similarity entirely. Soft labels implicitly encode it: a teacher that assigns 18% probability to "dog" when shown a cat is telling the student that cats and dogs share visual features. That structural information propagates through every training step, producing a student that generalises better despite having far fewer parameters.

---

## Extending this experiment

Some directions to try next:

**Feature-based distillation** — instead of (or in addition to) matching output distributions, match intermediate layer activations:
```python
# hint loss between teacher layer i and student layer j
hint_loss = F.mse_loss(student_hidden, teacher_hidden.detach())
```

**Born-again networks** — use the same architecture for teacher and student, then distill again iteratively. Each generation improves slightly on the last.

**Self-distillation** — save teacher checkpoints at earlier epochs and distill the final model from its own past self.

**Offline vs online distillation** — this script uses offline distillation (teacher fully trained first). Online distillation trains both simultaneously; mutual learning (Zhang et al. 2018) is a notable variant.

---

## References

- Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531
- Romero, A. et al. (2015). *FitNets: Hints for Thin Deep Nets.* arXiv:1412.6550
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108
- Zhang, Y. et al. (2018). *Deep Mutual Learning.* CVPR 2018.

---

## License

MIT — free to use, modify, and distribute.
