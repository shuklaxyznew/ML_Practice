# simple_vit — Vision Transformer from First Principles

A clean, heavily-commented PyTorch implementation of the Vision Transformer (ViT) architecture from the paper **"An Image is Worth 16×16 Words"** (Dosovitskiy et al., 2020). Built for learning and experimentation — every component maps directly to the theory.

---

## What's inside

```
simple_vit.py
├── PatchEmbedding          # image → sequence of patch vectors
├── PositionalEncoding      # inject spatial position into each token
├── MultiHeadSelfAttention  # every patch attends to every other patch
├── FeedForward             # per-token MLP (Linear → GELU → Linear)
├── TransformerBlock        # MHSA + FFN with LayerNorm + residuals
├── ViT                     # full model: embed → encode → classify
└── vit_tiny/small/base/large  # paper-standard configs
```

---

## Requirements

```bash
pip install torch einops
```

- **torch** ≥ 2.0 recommended (works on CPU and GPU)
- **einops** — makes tensor reshaping readable (`rearrange`, etc.)

---

## Quick start

```bash
# Run the built-in smoke test
python simple_vit.py
```

Expected output:

```
============================================================
Vision Transformer — smoke test
============================================================
Device: cpu

Total trainable parameters: 4,214,026
  ≈ 4.2M parameters

Input shape:  (4, 3, 224, 224)
Output shape: (4, 10)
  → batch_size=4, num_classes=10

Number of patches: 196  (= (224/16)² = 196)

Extracting attention maps...
  4 blocks × shape (1, 3, 197, 197)
  (batch=1, heads=3, tokens=197, tokens=197)

Verifying [CLS] token position: index 0 in the sequence ✓

Verifying gradient flow...
  Loss: 2.3041
  NaN gradients: False
  Gradient flow: ✓ OK

Done. Import ViT from this file to use in your project.
```

---

## Usage

### Custom model

```python
from simple_vit import ViT

model = ViT(
    image_size  = 224,    # H = W of input image
    patch_size  = 16,     # each patch is 16×16 pixels
    in_channels = 3,      # RGB
    num_classes = 10,     # e.g. CIFAR-10
    embed_dim   = 384,    # token dimension D
    depth       = 6,      # number of transformer blocks
    num_heads   = 6,      # attention heads (must divide embed_dim)
    mlp_ratio   = 4.0,    # FFN hidden dim = embed_dim × mlp_ratio
    dropout     = 0.1,
)

import torch
x = torch.randn(8, 3, 224, 224)   # batch of 8 images
logits = model(x)                  # (8, 10)
```

### Paper-standard configs

```python
from simple_vit import vit_tiny, vit_small, vit_base, vit_large

model = vit_tiny(num_classes=1000)   #  ~5M params  — fast experiments
model = vit_small(num_classes=1000)  # ~22M params  — accuracy/speed trade-off
model = vit_base(num_classes=1000)   # ~86M params  — ViT-B/16 from the paper
model = vit_large(num_classes=1000)  # ~307M params — ViT-L/16

print(model.get_num_params())
```

### Extract attention maps

```python
# Visualize what the model attends to
attn_maps = model.get_attention_maps(x[:1])  # single image

# attn_maps: list of L tensors, each (1, num_heads, N+1, N+1)
# Row 0 = attention FROM [CLS] token TO all patches — great for visualization

import matplotlib.pyplot as plt

layer_idx = 0
head_idx  = 0
cls_attn  = attn_maps[layer_idx][0, head_idx, 0, 1:]  # (N,) — CLS → patches
cls_attn  = cls_attn.reshape(14, 14)                   # reshape to spatial grid

plt.imshow(cls_attn.numpy(), cmap='hot')
plt.title(f'CLS attention — layer {layer_idx}, head {head_idx}')
plt.colorbar()
plt.show()
```

### Training on CIFAR-10 (minimal example)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from simple_vit import ViT

# Data
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
train_data = datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ViT(
    image_size=224, patch_size=16, num_classes=10,
    embed_dim=192, depth=6, num_heads=3, dropout=0.1
).to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f'Epoch {epoch+1}: loss={loss.item():.4f}')
```

> **Tip:** ViTs are data-hungry. For CIFAR-10, consider using a pretrained ViT-Base from `timm` and fine-tuning, or use DeiT which adds distillation to work well on smaller datasets.

---

## Architecture overview

```
Input image (B, 3, H, W)
        │
        ▼
 PatchEmbedding           # Conv2d(stride=P) → rearrange → (B, N, D)
        │                   N = (H/P)²  patches,  D = embed_dim
        ▼
 Prepend [CLS] token      # (B, N+1, D)
        │
        ▼
 PositionalEncoding       # learned (1, N+1, D), added in-place
        │
        ▼
 ┌─────────────────────┐
 │  TransformerBlock   │  ×L
 │  ┌───────────────┐  │
 │  │  LayerNorm    │  │
 │  │  MHSA         │  │   h heads, each dim D/h
 │  │  + residual   │  │   attention: O(N²) per head
 │  ├───────────────┤  │
 │  │  LayerNorm    │  │
 │  │  FFN          │  │   Linear(D→4D) → GELU → Linear(4D→D)
 │  │  + residual   │  │
 │  └───────────────┘  │
 └─────────────────────┘
        │
        ▼
 LayerNorm (final)
        │
        ▼
 Extract [CLS] output     # (B, D) — global image representation
        │
        ▼
 Linear head              # (B, num_classes)
        │
        ▼
 Class logits
```

---

## Model configs (paper values)

| Config     | `embed_dim` | `depth` | `num_heads` | Params | Notes |
|------------|-------------|---------|-------------|--------|-------|
| ViT-Tiny   | 192         | 12      | 3           | ~5M    | Fast experiments |
| ViT-Small  | 384         | 12      | 6           | ~22M   | Good trade-off |
| ViT-Base   | 768         | 12      | 12          | ~86M   | Standard ViT-B/16 |
| ViT-Large  | 1024        | 24      | 16          | ~307M  | High accuracy |

All configs use `patch_size=16`, `image_size=224`, `mlp_ratio=4.0`.

---

## Key design decisions explained

### Why `Conv2d` for patch embedding?

A `Conv2d(in_channels, D, kernel_size=P, stride=P)` with no overlap is mathematically identical to unfolding the image into patches and applying a shared linear layer. It's faster, uses less memory, and requires no explicit reshape before the projection.

### Why learnable 1D positional encodings?

The original ViT paper tested fixed sinusoidal 2D encodings, learned 1D, and learned 2D encodings — and found **no significant difference** in accuracy. Learned 1D is the simplest so it's the default here. After training, the learned position vectors automatically develop 2D spatial structure.

### Why pre-LN (LayerNorm before the sub-layer)?

Post-LN (the original transformer) is harder to train — gradients can vanish near the start of training. Pre-LN puts LayerNorm on the residual branch, keeping the main path clean and making gradient flow more stable. All modern ViT implementations use pre-LN.

### Why GELU instead of ReLU in the FFN?

GELU (Gaussian Error Linear Unit) is a smooth approximation to ReLU that works better empirically in transformer FFNs. It was introduced in BERT and adopted by nearly all subsequent transformer models.

---

## Extending this implementation

### Add stochastic depth (DropPath)

Randomly drops entire residual paths during training — a strong regularizer for ViTs:

```python
# In TransformerBlock.__init__:
from timm.layers import DropPath
self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

# In TransformerBlock.forward:
x = x + self.drop_path(self.attn(self.norm1(x)))
x = x + self.drop_path(self.ffn(self.norm2(x)))
```

### Use a different patch size

Smaller patches = more tokens = more compute but finer detail:

```python
model = ViT(patch_size=8, ...)   # 784 patches for 224×224 — expensive!
model = ViT(patch_size=32, ...)  # 49 patches — fast but coarser
```

### Global average pooling instead of [CLS]

Some variants average all patch outputs instead of using a [CLS] token:

```python
# Replace the CLS extraction in ViT.forward:
# cls_output = x[:, 0]
pooled = x[:, 1:].mean(dim=1)   # average over all patch tokens
logits = self.head(pooled)
```

---

## What to read next

| Paper | Key contribution |
|-------|-----------------|
| ViT (Dosovitskiy 2020) | This implementation — pure transformer for images |
| DeiT (Touvron 2021) | Distillation token; fixes ViT's data hunger |
| Swin Transformer (Liu 2021) | Hierarchical windows; O(N) attention |
| MAE (He 2022) | Masked autoencoder pretraining for ViT |
| DINO (Caron 2021) | Self-supervised ViT; attention maps become segmentation |

---

## References

```
@article{dosovitskiy2020image,
  title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author  = {Dosovitskiy, Alexey and others},
  journal = {ICLR 2021},
  year    = {2020}
}
```

---

## License

MIT — use freely for learning, research, and production.
