"""
Per-Image Disentanglement Story.

For each selected airplane image shows 5 columns:

  Col 1: Original airplane image
  Col 2: Full VAE reconstruction  Decoder(S, U)      -- VAE fidelity check
  Col 3: Shared knowledge only    Decoder(S, 0)       -- what is PRESERVED (sky, shape)
  Col 4: Unique knowledge only    Decoder(0, U)       -- what is ERASED by DKF
  Col 5: Mixed counterfactual     Decoder(S_air, U_bird) -- airplane structure in bird context

Run from week5_analysis/:
  uv run python visualize_image_decomposition.py

Output: week5_analysis/results/image_decomposition.png
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_W34  = os.path.normpath(os.path.join(_HERE, '..', 'week3-4'))
sys.path.insert(0, _W34)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torchvision

from beta_vae import BetaVAE
from config import (
    LATENT_DIM, NUM_CLASSES, BETA, CLASS_NAMES,
    FORGET_CLASS, VAE_PRETRAIN_EPOCHS
)
from config import get_device
from data_utils import get_transforms

DATA_DIR       = os.path.normpath(os.path.join(_W34, '..', 'week1_baseline', 'data'))
CHECKPOINT_DIR = os.path.join(_W34, 'checkpoints')
RESULTS_DIR    = os.path.join(_HERE, 'results')

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)


def to_img(t):
    return (t.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def load_vae(device):
    path = os.path.join(CHECKPOINT_DIR,
                        f'vae_pretrained_e{VAE_PRETRAIN_EPOCHS}_b{int(BETA)}.pth')
    if not os.path.exists(path):
        sys.exit(f"[ERROR] VAE checkpoint not found: {path}")
    vae = BetaVAE(latent_dim_s=LATENT_DIM, latent_dim_u=LATENT_DIM,
                  num_classes=NUM_CLASSES, beta=BETA).to(device)
    vae.load_state_dict(torch.load(path, map_location=device))
    vae.eval()
    print(f"[VAE] Loaded <- {path}")
    return vae


def get_samples(class_idx, n, use_train=True):
    _, test_tf = get_transforms()
    ds  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=use_train,
                                       download=False, transform=test_tf)
    idx = np.where(np.array(ds.targets) == class_idx)[0][:n]
    loader = DataLoader(Subset(ds, idx), batch_size=n, shuffle=False)
    x, _ = next(iter(loader))
    return x


@torch.no_grad()
def build_columns(vae, x_air, x_bird, device):
    """
    Returns 5 tensors, each shape (N, 3, 32, 32):
      orig, full_recon, shared_only, unique_only, mixed
    """
    x_air  = x_air.to(device)
    x_bird = x_bird.to(device)
    n      = x_air.size(0)

    s_air, _ = vae.encoder_s(x_air)
    u_air, _ = vae.encoder_u(x_air)
    u_bird, _= vae.encoder_u(x_bird[:n])

    zeros_u = torch.zeros_like(u_air)
    zeros_s = torch.zeros_like(s_air)

    full_recon  = vae.decoder(s_air, u_air)          # Decoder(S, U)
    shared_only = vae.decoder(s_air, zeros_u)         # Decoder(S, 0)
    unique_only = vae.decoder(zeros_s, u_air)         # Decoder(0, U)
    mixed       = vae.decoder(s_air, u_bird[:n])      # Decoder(S_air, U_bird)

    return (x_air.cpu(), full_recon.cpu(),
            shared_only.cpu(), unique_only.cpu(), mixed.cpu())


def main():
    device = get_device()
    print(f"[Device] {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    vae   = load_vae(device)
    N_ROW = 4   # number of airplane images to show

    x_air  = get_samples(FORGET_CLASS, n=N_ROW)   # airplane
    x_bird = get_samples(2,            n=N_ROW)   # bird (for mixing)

    print("Building columns...")
    orig, full_recon, shared_only, unique_only, mixed = \
        build_columns(vae, x_air, x_bird, device)

    # ── Column definitions ─────────────────────────────────────────────────────
    col_data   = [orig, full_recon, shared_only, unique_only, mixed]
    col_titles = [
        'Original\nAirplane',
        'Full\nReconstruction\nDecoder(S + U)',
        'Shared Knowledge\nDecoder(S,  0)\n[PRESERVED by DKF]',
        'Unique Knowledge\nDecoder(0,  U)\n[ERASED by DKF]',
        'Mixed Counterfactual\nDecoder(S_airplane, U_bird)\n[Shared features in bird context]',
    ]
    col_colors = ['#2c3e50', '#7f8c8d', '#2980b9', '#8e44ad', '#27ae60']
    col_bg     = ['#ffffff', '#f8f9fa', '#ebf5fb', '#f5eef8', '#eafaf1']

    N_COLS = len(col_data)
    fig, axes = plt.subplots(N_ROW, N_COLS,
                             figsize=(N_COLS * 2.4, N_ROW * 2.4 + 1.4))
    fig.suptitle(
        'Per-Image Disentanglement:  What Does DKF Preserve vs Erase?\n'
        f'Forget class: Airplane  |  Mixed with: Bird  |  '
        f'β-VAE  (β={int(BETA)}, {VAE_PRETRAIN_EPOCHS} epochs)',
        fontsize=12, fontweight='bold', y=0.99
    )

    for col_i, (col_imgs, title, color, bg) in enumerate(
            zip(col_data, col_titles, col_colors, col_bg)):
        for row_i in range(N_ROW):
            ax = axes[row_i, col_i]
            ax.imshow(to_img(col_imgs[row_i]))
            ax.axis('off')
            ax.set_facecolor(bg)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)
                spine.set_visible(True)
            if row_i == 0:
                ax.set_title(title, fontsize=8, fontweight='bold',
                             color=color, pad=5, linespacing=1.4)

    # Row labels
    for row_i in range(N_ROW):
        axes[row_i, 0].set_ylabel(f'Image {row_i+1}', fontsize=8,
                                  color='#555', rotation=90, labelpad=4)

    # Bottom annotation
    fig.text(
        0.5, 0.005,
        'DKF targets the UNIQUE space (purple) for erasure.  '
        'The SHARED space (blue) is geometrically protected — '
        'its features survive in retain classes (bird, ship, etc.)',
        ha='center', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdfefe', alpha=0.9)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = os.path.join(RESULTS_DIR, 'image_decomposition.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {out}")


if __name__ == '__main__':
    main()
