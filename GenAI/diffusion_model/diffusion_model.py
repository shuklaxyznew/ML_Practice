"""
Simple Diffusion Model (DDPM) — from scratch with PyTorch
==========================================================
Covers: Data prep → Noise schedule → U-Net denoiser → Training → Sampling
Dataset: MNIST (28x28 grayscale) — runs on CPU or GPU
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    # Data
    image_size   = 28
    channels     = 1                  # grayscale
    batch_size   = 128
    data_dir     = "./data"

    # Diffusion
    T            = 1000               # total timesteps
    beta_start   = 1e-4
    beta_end     = 0.02
    schedule     = "cosine"           # "linear" | "cosine"

    # U-Net
    dim          = 64                 # base channel width
    dim_mults    = (1, 2, 4)          # channel multipliers per level
    time_emb_dim = 128

    # Training
    epochs       = 10
    lr           = 2e-4
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    sample_dir   = "./samples"
    ckpt_path    = "./ddpm_mnist.pt"

cfg = Config()
os.makedirs(cfg.sample_dir, exist_ok=True)
print(f"Device: {cfg.device}")


# ─────────────────────────────────────────────
# 2. DATA PREPARATION
# ─────────────────────────────────────────────
def get_dataloader(cfg: Config) -> DataLoader:
    """
    Load MNIST, normalize to [-1, 1] (standard for diffusion models).
    The model learns to denoise images in this range.
    """
    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),   # [0,1] → [-1,1]
    ])
    dataset = datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=cfg.batch_size,
                      shuffle=True, num_workers=2, pin_memory=True)


# ─────────────────────────────────────────────
# 3. NOISE SCHEDULE  (Forward Process q)
# ─────────────────────────────────────────────
class NoiseSchedule:
    """
    Precomputes α, ᾱ, β and all derived quantities for the
    closed-form forward process:
        q(xₜ | x₀) = N(xₜ ; √ᾱₜ · x₀ , (1−ᾱₜ)·I)
    """
    def __init__(self, cfg: Config):
        T = cfg.T
        if cfg.schedule == "linear":
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, T)
        elif cfg.schedule == "cosine":
            # Improved DDPM (Nichol & Dhariwal 2021) cosine schedule
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0, 0.999).float()
        else:
            raise ValueError(f"Unknown schedule: {cfg.schedule}")

        alphas            = 1.0 - betas
        alphas_cumprod    = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store everything we'll need
        self.T                        = T
        self.betas                    = betas
        self.alphas_cumprod           = alphas_cumprod
        self.sqrt_alphas_cumprod      = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        # For reverse process posterior q(x_{t-1} | xₜ, x₀)
        self.posterior_variance       = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.posterior_mean_coef1     = (
            betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        )
        self.posterior_mean_coef2     = (
            (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)
        )

    def _gather(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Index into a 1-D schedule tensor using batch timestep indices."""
        return tensor.to(t.device)[t][:, None, None, None]   # (B,1,1,1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion: add noise at timestep t.
        xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_abar   = self._gather(self.sqrt_alphas_cumprod, t)
        sqrt_1mabar = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return sqrt_abar * x0 + sqrt_1mabar * noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor,
                   eps: torch.Tensor) -> torch.Tensor:
        """Reconstruct x₀ from xₜ and predicted noise ε̂."""
        sqrt_abar   = self._gather(self.sqrt_alphas_cumprod, t)
        sqrt_1mabar = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return (xt - sqrt_1mabar * eps) / sqrt_abar

    def p_mean_variance(self, model: nn.Module, xt: torch.Tensor,
                        t: torch.Tensor) -> tuple:
        """
        Reverse process: compute mean & variance for p(x_{t-1} | xₜ).
        Uses the reparameterization: predict ε then compute mean.
        """
        eps_pred  = model(xt, t)
        x0_pred   = self.predict_x0(xt, t, eps_pred).clamp(-1, 1)
        mean = (
            self._gather(self.posterior_mean_coef1, t) * x0_pred +
            self._gather(self.posterior_mean_coef2, t) * xt
        )
        var = self._gather(self.posterior_variance, t)
        return mean, var


# ─────────────────────────────────────────────
# 4. DENOISING NETWORK  (U-Net)
# ─────────────────────────────────────────────

# ---- Sinusoidal time embedding ----
class SinusoidalPositionEmbeddings(nn.Module):
    """Encode scalar timestep t into a rich feature vector."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args  = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


# ---- Residual block with time conditioning ----
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.norm2  = nn.GroupNorm(8, out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = (nn.Conv2d(in_ch, out_ch, 1)
                       if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ---- Simple self-attention ----
class Attention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads   = heads
        self.norm    = nn.GroupNorm(8, ch)
        self.to_qkv  = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.to_out  = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v  = self.to_qkv(h).chunk(3, dim=1)

        # Reshape to (B·heads, head_dim, HW)
        head_dim = C // self.heads
        reshape  = lambda t: t.reshape(B * self.heads, head_dim, H * W)
        q, k, v  = map(reshape, (q, k, v))

        scale  = head_dim ** -0.5
        attn   = torch.softmax((q.transpose(1, 2) @ k) * scale, dim=-1)
        out    = (attn @ v.transpose(1, 2)).transpose(1, 2)
        out    = out.reshape(B, C, H, W)
        return x + self.to_out(out)


# ---- Full U-Net ----
class UNet(nn.Module):
    """
    Encoder → Bottleneck → Decoder with skip connections.
    Time embedding is injected at every residual block.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        ch   = cfg.dim
        mults = cfg.dim_mults
        dims  = [ch * m for m in mults]           # e.g. [64, 128, 256]
        T_emb = cfg.time_emb_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(ch),
            nn.Linear(ch, T_emb),
            nn.SiLU(),
            nn.Linear(T_emb, T_emb),
        )

        # Initial projection
        self.init_conv = nn.Conv2d(cfg.channels, ch, 7, padding=3)

        # ── Encoder ──
        self.downs = nn.ModuleList()
        in_ch = ch
        for i, out_ch in enumerate(dims):
            is_last = (i == len(dims) - 1)
            self.downs.append(nn.ModuleList([
                ResBlock(in_ch, out_ch, T_emb),
                ResBlock(out_ch, out_ch, T_emb),
                Attention(out_ch),
                nn.Conv2d(out_ch, out_ch, 4, 2, 1) if not is_last
                else nn.Identity(),   # downsample except last level
            ]))
            in_ch = out_ch

        # ── Bottleneck ──
        mid_ch = dims[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, T_emb)
        self.mid_attn   = Attention(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, T_emb)

        # ── Decoder ──
        self.ups = nn.ModuleList()
        for i, out_ch in enumerate(reversed(dims)):
            is_last = (i == len(dims) - 1)
            in_ch_up = out_ch * 2    # skip connection doubles channels
            self.ups.append(nn.ModuleList([
                ResBlock(in_ch_up, out_ch, T_emb),
                ResBlock(out_ch, out_ch, T_emb),
                Attention(out_ch),
                nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1) if not is_last
                else nn.Identity(),  # upsample except last level
            ]))

        self.final_norm = nn.GroupNorm(8, ch)
        self.final_conv = nn.Conv2d(ch, cfg.channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)           # (B, T_emb)

        x = self.init_conv(x)
        skips = []

        # Encoder
        for res1, res2, attn, down in self.downs:
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Decoder
        for res1, res2, attn, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = attn(x)
            x = up(x)

        return self.final_conv(F.silu(self.final_norm(x)))


# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────

def train(cfg: Config):
    loader   = get_dataloader(cfg)
    schedule = NoiseSchedule(cfg)
    model    = UNet(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params/1e6:.2f}M")

    loss_history = []

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0

        for step, (x0, _) in enumerate(loader):
            x0 = x0.to(cfg.device)
            B  = x0.size(0)

            # ── Sample random timesteps ──
            t = torch.randint(0, cfg.T, (B,), device=cfg.device)

            # ── Forward process: add noise ──
            noise  = torch.randn_like(x0)
            xt     = schedule.q_sample(x0, t, noise)

            # ── Predict the noise ──
            eps_pred = model(xt, t)

            # ── Simple MSE loss (equivalent to VLB up to constant) ──
            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if step % 100 == 0:
                print(f"  Epoch {epoch+1}/{cfg.epochs} | "
                      f"Step {step}/{len(loader)} | "
                      f"Loss {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Save samples each epoch
        model.eval()
        with torch.no_grad():
            samples = sample_ddpm(model, schedule, cfg, n=16)
        save_image(
            samples, f"{cfg.sample_dir}/epoch_{epoch+1:03d}.png",
            nrow=4, normalize=True, value_range=(-1, 1)
        )

    # Save checkpoint
    torch.save(model.state_dict(), cfg.ckpt_path)
    print(f"Model saved → {cfg.ckpt_path}")

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, cfg.epochs + 1), loss_history, marker='o', color='royalblue')
    plt.title("DDPM Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.sample_dir}/loss_curve.png", dpi=150)
    plt.close()
    print(f"Loss curve saved → {cfg.sample_dir}/loss_curve.png")

    return model, schedule, loss_history


# ─────────────────────────────────────────────
# 6. SAMPLING  (Reverse Process)
# ─────────────────────────────────────────────

@torch.no_grad()
def sample_ddpm(model: nn.Module, schedule: NoiseSchedule,
                cfg: Config, n: int = 16) -> torch.Tensor:
    """
    DDPM ancestral sampling (Ho et al. 2020).
    Start from pure noise xT and iteratively denoise T → 0.
    """
    model.eval()
    x = torch.randn(n, cfg.channels, cfg.image_size, cfg.image_size,
                    device=cfg.device)

    for t_val in reversed(range(cfg.T)):
        t = torch.full((n,), t_val, device=cfg.device, dtype=torch.long)
        mean, var = schedule.p_mean_variance(model, x, t)

        noise = torch.randn_like(x) if t_val > 0 else torch.zeros_like(x)
        x = mean + var.sqrt() * noise

    return x.clamp(-1, 1)


@torch.no_grad()
def sample_ddim(model: nn.Module, schedule: NoiseSchedule,
                cfg: Config, n: int = 16,
                ddim_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
    """
    DDIM deterministic sampler (Song et al. 2020).
    Runs in `ddim_steps` steps instead of T — much faster!
    eta=0 → deterministic; eta=1 → stochastic (≈ DDPM)
    """
    model.eval()
    step_size  = cfg.T // ddim_steps
    timesteps  = list(reversed(range(0, cfg.T, step_size)))

    x = torch.randn(n, cfg.channels, cfg.image_size, cfg.image_size,
                    device=cfg.device)

    abar = schedule.alphas_cumprod.to(cfg.device)

    for i, t_val in enumerate(timesteps):
        t      = torch.full((n,), t_val, device=cfg.device, dtype=torch.long)
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0

        eps    = model(x, t)
        a_t    = abar[t_val]
        a_prev = abar[t_prev]

        x0_pred = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        sigma   = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()
        noise   = torch.randn_like(x)

        x = a_prev.sqrt() * x0_pred + (1 - a_prev - sigma**2).sqrt() * eps + sigma * noise

    return x.clamp(-1, 1)


# ─────────────────────────────────────────────
# 7. VISUALISE FORWARD PROCESS
# ─────────────────────────────────────────────

def visualise_forward_process(schedule: NoiseSchedule, cfg: Config):
    """Show a single image corrupted at t = 0, 100, 300, 500, 700, 1000."""
    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root=cfg.data_dir, train=True, download=True,
                             transform=transform)
    x0 = dataset[0][0].unsqueeze(0)  # (1,1,28,28)

    steps  = [0, 100, 300, 500, 700, 999]
    fig, axes = plt.subplots(1, len(steps), figsize=(14, 3))
    for ax, t_val in zip(axes, steps):
        t  = torch.tensor([t_val])
        xt = schedule.q_sample(x0, t)
        img = xt[0, 0].numpy()
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"t={t_val}")
        ax.axis("off")
    plt.suptitle("Forward Diffusion Process  q(xₜ | x₀)", y=1.02)
    plt.tight_layout()
    path = f"{cfg.sample_dir}/forward_process.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Forward process viz saved → {path}")


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    schedule = NoiseSchedule(cfg)

    # Visualise noise corruption before training
    visualise_forward_process(schedule, cfg)

    # Train
    model, schedule, loss_history = train(cfg)

    # Final DDPM samples
    print("Generating final DDPM samples …")
    samples = sample_ddpm(model, schedule, cfg, n=64)
    save_image(samples, f"{cfg.sample_dir}/final_ddpm.png",
               nrow=8, normalize=True, value_range=(-1, 1))

    # Final DDIM samples (50 steps)
    print("Generating final DDIM samples (50 steps) …")
    samples_ddim = sample_ddim(model, schedule, cfg, n=64, ddim_steps=50)
    save_image(samples_ddim, f"{cfg.sample_dir}/final_ddim.png",
               nrow=8, normalize=True, value_range=(-1, 1))

    print("\nDone! Check the ./samples directory for outputs.")
