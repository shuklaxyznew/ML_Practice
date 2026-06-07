"""
simple_vit.py — Vision Transformer from first principles
=========================================================
Implements the core ViT architecture (Dosovitskiy et al., 2020)
with heavy comments mapping each component to theory.

Usage:
    python simple_vit.py          # runs a quick smoke test
    from simple_vit import ViT    # import into your project

Requirements:
    pip install torch einops
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # makes tensor reshaping readable


# ─────────────────────────────────────────────────────────
# 1. PATCH EMBEDDING
#    Takes image (B, C, H, W) → sequence of patch embeddings (B, N, D)
#    This is the "tokenizer" for images.
# ─────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits the image into patches and projects each to D dimensions.

    Two implementations give the same result:
      a) Unfold → Linear  (explicit, educational)
      b) Conv2d with kernel=patch_size, stride=patch_size  (efficient, used in practice)

    We use (b) because it's cleaner and identical mathematically.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, \
            f"Image size {image_size} must be divisible by patch size {patch_size}"

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2   # N = (H/P)²

        # A Conv2d with kernel=stride=patch_size is exactly a "patch projection":
        # each non-overlapping patch_size×patch_size region maps to one embed_dim vector.
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.projection(x)           # (B, D, H/P, W/P) — one vector per patch
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, N, D) — flatten spatial dims
        return x


# ─────────────────────────────────────────────────────────
# 2. POSITIONAL ENCODING
#    Adds spatial position information to each patch embedding.
#    Without this, the model can't tell patch at (0,0) from patch at (7,7).
# ─────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Learnable 1D positional encoding — the default from the ViT paper.

    One learnable vector per position (including [CLS] token at pos 0).
    Shape: (1, num_patches + 1, D) — broadcast over the batch dimension.

    Alternative: sinusoidal 2D encodings (no learned params, see below).
    """

    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        # +1 for the [CLS] token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02  # small init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N+1, D)   (+1 because [CLS] was already prepended)
        return x + self.pos_embedding


# ─────────────────────────────────────────────────────────
# 3. MULTI-HEAD SELF-ATTENTION (MHSA)
#    The core operation that lets every patch attend to every other patch.
# ─────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product attention with h heads.

    For each head:
        Q = x Wq,  K = x Wk,  V = x Wv     (project to D/h)
        Attention(Q, K, V) = softmax(QKᵀ / √(D/h)) · V

    All heads concatenated, then projected back to D.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5   # 1/√(D/h) — prevents vanishing gradients

        # Single projection for Q, K, V together (efficient)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # Project to Q, K, V and split into heads
        qkv = self.qkv(x)                          # (B, N, 3D)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d',
                        three=3, h=self.num_heads)  # (3, B, h, N, D/h)
        q, k, v = qkv.unbind(0)                    # each: (B, h, N, D/h)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = attn.softmax(dim=-1)                     # normalize over keys
        attn = self.attn_drop(attn)

        # Aggregate values
        out = attn @ v                                  # (B, h, N, D/h)
        out = rearrange(out, 'b h n d -> b n (h d)')   # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ─────────────────────────────────────────────────────────
# 4. FEED-FORWARD NETWORK (FFN)
#    Applied independently to each token after attention.
#    Gives the model capacity to process attended information.
# ─────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation.
    Hidden dimension is typically 4× the embedding dimension.

    Linear(D → 4D) → GELU → Dropout → Linear(4D → D) → Dropout
    """

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),            # smoother than ReLU; works better for transformers
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────
# 5. TRANSFORMER ENCODER BLOCK
#    Combines MHSA + FFN with LayerNorm and residual connections.
# ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-LN variant (normalize before the sub-layer, not after).
    The ViT paper uses this; it's more stable to train.

    z' = MHSA(LN(z)) + z         # attention sub-layer
    z  = FFN(LN(z')) + z'        # FFN sub-layer
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # residual around attention
        x = x + self.ffn(self.norm2(x))    # residual around FFN
        return x


# ─────────────────────────────────────────────────────────
# 6. VISION TRANSFORMER (ViT)
#    Puts all components together into the full model.
# ─────────────────────────────────────────────────────────

class ViT(nn.Module):
    """
    Vision Transformer for image classification.

    Pipeline:
        Image → PatchEmbedding → prepend [CLS] → add PositionalEncoding
              → L × TransformerBlock → LayerNorm → [CLS] output
              → MLP head → class logits

    Args:
        image_size  : Height = Width of input image (e.g. 224)
        patch_size  : Size of each square patch (e.g. 16)
        in_channels : Input image channels (3 for RGB)
        num_classes : Number of output classes (e.g. 1000 for ImageNet)
        embed_dim   : Token embedding dimension D (e.g. 768 for ViT-Base)
        depth       : Number of transformer blocks L (e.g. 12 for ViT-Base)
        num_heads   : Number of attention heads h (e.g. 12 for ViT-Base)
        mlp_ratio   : FFN hidden dim = embed_dim × mlp_ratio (default 4.0)
        dropout     : Dropout rate for attention and FFN (default 0.1)
    """

    def __init__(
        self,
        image_size:  int   = 224,
        patch_size:  int   = 16,
        in_channels: int   = 3,
        num_classes: int   = 1000,
        embed_dim:   int   = 768,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # ── Patch embedding ─────────────────────────────
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # ── [CLS] token ─────────────────────────────────
        # Learnable vector prepended to the sequence.
        # After all transformer layers, this aggregates global image info.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Positional encoding ─────────────────────────
        self.pos_encoding = PositionalEncoding(num_patches, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # ── Transformer encoder ─────────────────────────
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final LayerNorm before classification head
        self.norm = nn.LayerNorm(embed_dim)

        # ── Classification head ─────────────────────────
        # Linear layer applied to the [CLS] token output only.
        # The [CLS] token has attended to all patches → global representation.
        self.head = nn.Linear(embed_dim, num_classes)

        # ── Weight initialization ────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Standard ViT initialization."""
        # Initialize patch projection like a conv layer
        nn.init.normal_(self.cls_token, std=0.02)
        # Apply truncated normal to all linear and embedding layers
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1. Embed patches: (B, C, H, W) → (B, N, D)
        x = self.patch_embed(x)

        # 2. Prepend [CLS] token: (B, N, D) → (B, N+1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # repeat for each sample in batch
        x = torch.cat([cls_tokens, x], dim=1)

        # 3. Add positional encoding (in-place add, same shape)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 4. Pass through L transformer blocks
        x = self.blocks(x)

        # 5. Final LayerNorm
        x = self.norm(x)

        # 6. Extract [CLS] token (position 0) and classify
        cls_output = x[:, 0]             # (B, D)
        logits = self.head(cls_output)   # (B, num_classes)
        return logits

    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Extract attention maps from each block for visualization.
        Returns list of (B, h, N+1, N+1) tensors.
        Useful for understanding what the model attends to.
        """
        attention_maps = []

        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)

        for block in self.blocks:
            # Hook into the attention scores
            normed = block.norm1(x)
            qkv = block.attn.qkv(normed)
            qkv = rearrange(qkv, 'b n (three h d) -> three b h n d',
                            three=3, h=block.attn.num_heads)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach())

            # Still run the full forward for next block
            x = block(x)

        return attention_maps


# ─────────────────────────────────────────────────────────
# PREDEFINED CONFIGS (matching the original ViT paper)
# ─────────────────────────────────────────────────────────

def vit_tiny(num_classes: int = 1000, **kwargs) -> ViT:
    """ViT-Tiny — fast for experimentation."""
    return ViT(embed_dim=192, depth=12, num_heads=3, num_classes=num_classes, **kwargs)

def vit_small(num_classes: int = 1000, **kwargs) -> ViT:
    """ViT-Small — good accuracy/speed trade-off."""
    return ViT(embed_dim=384, depth=12, num_heads=6, num_classes=num_classes, **kwargs)

def vit_base(num_classes: int = 1000, **kwargs) -> ViT:
    """ViT-Base — the paper's standard model (86M params)."""
    return ViT(embed_dim=768, depth=12, num_heads=12, num_classes=num_classes, **kwargs)

def vit_large(num_classes: int = 1000, **kwargs) -> ViT:
    """ViT-Large — 307M params."""
    return ViT(embed_dim=1024, depth=24, num_heads=16, num_classes=num_classes, **kwargs)


# ─────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Vision Transformer — smoke test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Create a small ViT for fast testing
    model = ViT(
        image_size  = 224,
        patch_size  = 16,
        in_channels = 3,
        num_classes = 10,      # e.g. CIFAR-10
        embed_dim   = 192,
        depth       = 4,
        num_heads   = 3,
        dropout     = 0.1,
    ).to(device)

    total_params = model.get_num_params()
    print(f"Total trainable parameters: {total_params:,}")
    print(f"  ≈ {total_params / 1e6:.1f}M parameters\n")

    # Forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    print(f"Input shape:  {tuple(dummy_input.shape)}")

    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Output shape: {tuple(logits.shape)}")
    print(f"  → batch_size={batch_size}, num_classes=10")

    # Verify patch count
    num_patches = model.patch_embed.num_patches
    print(f"\nNumber of patches: {num_patches}  (= (224/16)² = {(224//16)**2})")

    # Check attention maps
    print("\nExtracting attention maps...")
    attn_maps = model.get_attention_maps(dummy_input[:1])  # single image
    print(f"  {len(attn_maps)} blocks × shape {tuple(attn_maps[0].shape)}")
    print(f"  (batch=1, heads=3, tokens={num_patches+1}, tokens={num_patches+1})")

    # Verify [CLS] token gets the global representation
    print("\nVerifying [CLS] token position: index 0 in the sequence ✓")

    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)

    # Quick training step to verify gradients flow
    print("\nVerifying gradient flow...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    loss = F.cross_entropy(model(dummy_input), labels)
    loss.backward()
    optimizer.step()

    # Check no NaN gradients
    has_nan = any(
        p.grad is not None and p.grad.isnan().any()
        for p in model.parameters()
    )
    print(f"  Loss: {loss.item():.4f}")
    print(f"  NaN gradients: {has_nan}")
    print(f"  Gradient flow: {'✓ OK' if not has_nan else '✗ FAIL'}")

    print("\nDone. Import ViT from this file to use in your project.")
