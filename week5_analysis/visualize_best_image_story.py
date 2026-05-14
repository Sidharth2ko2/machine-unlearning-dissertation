"""
Best-Image Disentanglement Story.

Picks the sharpest/most recognizable airplane and bird images from CIFAR-10,
then shows the full disentanglement story in a clean presentation layout:

  [Original (upscaled)]  |  [Full Recon]  |  [Shared S]  |  [Unique U]  |  [Mixed S_air+U_bird]

- Original displayed at 4x upscale (LANCZOS) so it looks clear in slides
- VAE operates on the actual 32x32 CIFAR data (no domain shift)
- Outputs also upscaled 4x for visual clarity

Run from week5_analysis/:
  uv run python visualize_best_image_story.py

Output: week5_analysis/results/best_image_story.png
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
from matplotlib.gridspec import GridSpec
from PIL import Image
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

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
UPSCALE = 8   # display 32x32 as 256x256


def unnorm(t):
    """Remove CIFAR normalisation -> [0,1]."""
    return (t.cpu() * STD + MEAN).clamp(0, 1)


def to_pil_upscaled(tensor_chw):
    """Convert (C,H,W) tensor to a PIL image upscaled UPSCALE×."""
    arr = (unnorm(tensor_chw).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    return pil.resize((arr.shape[1] * UPSCALE, arr.shape[0] * UPSCALE),
                      resample=Image.LANCZOS)


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


def pick_sharpest(class_idx, top_n=4, use_train=True):
    """
    Return the top_n sharpest images for a class,
    measured by pixel std-dev (high contrast = more recognisable).
    """
    _, test_tf = get_transforms()
    ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=use_train,
                                      download=False, transform=test_tf)
    idx_all = np.where(np.array(ds.targets) == class_idx)[0][:500]
    loader  = DataLoader(Subset(ds, idx_all), batch_size=256, shuffle=False)
    imgs    = torch.cat([x for x, _ in loader])          # (N, 3, 32, 32)

    # Score by std-dev across spatial dims (sharpness proxy)
    scores  = imgs.std(dim=[1, 2, 3])
    top_idx = scores.argsort(descending=True)[:top_n]
    return imgs[top_idx]


@torch.no_grad()
def decompose(vae, x_air, x_bird, device):
    x_air  = x_air.to(device)
    x_bird = x_bird.to(device)
    n      = x_air.size(0)

    s_air,  _ = vae.encoder_s(x_air)
    u_air,  _ = vae.encoder_u(x_air)
    u_bird, _ = vae.encoder_u(x_bird[:n])

    zeros_u = torch.zeros_like(u_air)
    zeros_s = torch.zeros_like(s_air)

    return dict(
        orig        = x_air.cpu(),
        full_recon  = vae.decoder(s_air, u_air).cpu(),
        shared_only = vae.decoder(s_air, zeros_u).cpu(),
        unique_only = vae.decoder(zeros_s, u_air).cpu(),
        mixed       = vae.decoder(s_air, u_bird[:n]).cpu(),
    )


def main():
    device = get_device()
    print(f"[Device] {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    vae = load_vae(device)

    print("Picking sharpest airplane images...")
    x_air  = pick_sharpest(FORGET_CLASS, top_n=3)
    print("Picking sharpest bird images...")
    x_bird = pick_sharpest(2, top_n=3)

    cols = decompose(vae, x_air, x_bird, device)

    N_ROWS = x_air.size(0)
    N_COLS = 5

    col_keys   = ['orig', 'full_recon', 'shared_only', 'unique_only', 'mixed']
    col_titles = [
        'Original Airplane\n(CIFAR-10)',
        'Full Reconstruction\nDecoder( S + U )',
        'Shared Knowledge\nDecoder( S,  0 )\n← PRESERVED by DKF',
        'Unique Knowledge\nDecoder( 0,  U )\n← ERASED by DKF',
        'Mixed Counterfactual\nDecoder( S_airplane,  U_bird )\n← Shared features survive in new class',
    ]
    col_colors = ['#1a1a2e', '#636e72', '#2980b9', '#8e44ad', '#27ae60']
    header_bg  = ['#dfe6e9', '#f0f0f0', '#d6eaf8', '#e8daef', '#d5f5e3']

    fig_w = N_COLS * 2.8
    fig_h = N_ROWS * 2.8 + 1.6
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(fig_w, fig_h),
                             gridspec_kw={'wspace': 0.06, 'hspace': 0.08})

    fig.suptitle(
        'Disentangled Machine Unlearning — DKF on Airplane (Forget Class)\n'
        'What is preserved (Shared S) vs erased (Unique U)?',
        fontsize=13, fontweight='bold', y=1.00
    )

    for col_i, (key, title, color, bg) in enumerate(
            zip(col_keys, col_titles, col_colors, header_bg)):
        for row_i in range(N_ROWS):
            ax  = axes[row_i, col_i]
            img = to_pil_upscaled(cols[key][row_i])
            ax.imshow(img)
            ax.axis('off')
            # Coloured border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
            if row_i == 0:
                ax.set_title(title, fontsize=8, fontweight='bold', color=color,
                             pad=6, linespacing=1.5,
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor=bg, alpha=0.85))
        axes[0, col_i].set_title(title, fontsize=8, fontweight='bold',
                                  color=color, pad=6, linespacing=1.5,
                                  bbox=dict(boxstyle='round,pad=0.3',
                                            facecolor=bg, alpha=0.85))

    # Row labels
    for row_i in range(N_ROWS):
        axes[row_i, 0].set_ylabel(f'Airplane #{row_i+1}',
                                   fontsize=8.5, color='#444',
                                   rotation=90, labelpad=6)

    fig.text(
        0.5, 0.005,
        'S-space (blue) = structural context shared across classes  |  '
        'U-space (purple) = class-specific features  |  '
        'DKF erases U, preserves S  |  Mixed column shows S carries airplane structure into bird context',
        ha='center', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', alpha=0.95)
    )

    out = os.path.join(RESULTS_DIR, 'best_image_story.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {out}")


if __name__ == '__main__':
    main()
