"""
Shared vs Unique Knowledge Decomposition Visualization.

For 2 classes (airplane = forget, bird = retain) shows a 3-row grid per class:

  Row 1: Original image  X
  Row 2: Shared reconstruction    X_S = Decoder(S_mu,  0)
          → what S-space captures: structural/contextual features (sky, shape)
  Row 3: Unique reconstruction    X_U = Decoder(0,     U_mu)
          → what U-space captures: class-specific features (landing gear, feathers)

This directly answers: "what is shared knowledge vs unique knowledge?"

Run from week5_analysis/:
  uv run python visualize_su_decomposition.py

Output: week5_analysis/results/su_decomposition.png
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


def to_display(t):
    return (t.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def load_vae(device):
    path = os.path.join(CHECKPOINT_DIR,
                        f'vae_pretrained_e{VAE_PRETRAIN_EPOCHS}_b{int(BETA)}.pth')
    if not os.path.exists(path):
        sys.exit(f"[ERROR] VAE checkpoint not found: {path}\n"
                 "Run:  cd ../week3-4 && uv run python run_experiments.py")
    vae = BetaVAE(latent_dim_s=LATENT_DIM, latent_dim_u=LATENT_DIM,
                  num_classes=NUM_CLASSES, beta=BETA).to(device)
    vae.load_state_dict(torch.load(path, map_location=device))
    vae.eval()
    print(f"[VAE] Loaded <- {path}")
    return vae


def get_samples(class_idx, n=6):
    _, test_tf = get_transforms()
    ds  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                       download=False, transform=test_tf)
    idx = np.where(np.array(ds.targets) == class_idx)[0][:n]
    loader = DataLoader(Subset(ds, idx), batch_size=n, shuffle=False)
    x, _ = next(iter(loader))
    return x


@torch.no_grad()
def decompose(vae, x, device):
    """
    Returns:
      x_s : Decoder(s_mu, zeros)  — shared-only reconstruction
      x_u : Decoder(zeros, u_mu)  — unique-only reconstruction
    """
    x    = x.to(device)
    s_mu, _ = vae.encoder_s(x)
    u_mu, _ = vae.encoder_u(x)

    zeros_u = torch.zeros_like(u_mu)
    zeros_s = torch.zeros_like(s_mu)

    x_s = vae.decoder(s_mu, zeros_u)   # shared knowledge only
    x_u = vae.decoder(zeros_s, u_mu)   # unique knowledge only
    return x.cpu(), x_s.cpu(), x_u.cpu()


def plot_class_panel(fig, outer_gs, col, class_idx, x, x_s, x_u, n_cols):
    """Draw a 3-row x n_cols panel for one class inside outer_gs[col]."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    class_name = CLASS_NAMES[class_idx]
    is_forget  = (class_idx == FORGET_CLASS)
    title_color = '#c0392b' if is_forget else '#27ae60'
    role        = 'Forget class' if is_forget else 'Retain class'

    inner = GridSpecFromSubplotSpec(3, n_cols, subplot_spec=outer_gs[col],
                                    hspace=0.06, wspace=0.04)

    row_data   = [x, x_s, x_u]
    row_labels = [
        f'Original\n({class_name})',
        'Shared (S)\nreconstruction\n→ structure, context',
        'Unique (U)\nreconstruction\n→ class-specific features',
    ]
    row_colors = ['#2c3e50', '#2980b9', '#8e44ad']

    for row_i, (imgs, label, color) in enumerate(
            zip(row_data, row_labels, row_colors)):
        for col_i in range(n_cols):
            ax = fig.add_subplot(inner[row_i, col_i])
            ax.imshow(to_display(imgs[col_i]))
            ax.axis('off')
            if col_i == 0:
                ax.set_ylabel(label, fontsize=7.5, color=color,
                              fontweight='bold', rotation=90, labelpad=6,
                              va='center')
            if row_i == 0 and col_i == n_cols // 2:
                ax.set_title(
                    f'{class_name.upper()}  ({role})',
                    fontsize=11, fontweight='bold', color=title_color, pad=8)


def main():
    device = get_device()
    print(f"[Device] {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    vae    = load_vae(device)
    N_COLS = 6

    # Classes: airplane (forget=0) and bird (retain=2)
    classes = [FORGET_CLASS, 2]

    print("\nDecomposing images into S and U reconstructions...")
    decompositions = {}
    for cls in classes:
        x = get_samples(cls, n=N_COLS)
        x_orig, x_s, x_u = decompose(vae, x, device)
        decompositions[cls] = (x_orig, x_s, x_u)
        print(f"  {CLASS_NAMES[cls]}: done")

    # ── Figure layout ──────────────────────────────────────────────────────────
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        'Disentangled Knowledge: Shared (S) vs Unique (U) Reconstruction\n'
        'β-VAE  |  CIFAR-10  |  Decoder(S, 0)  vs  Decoder(0, U)',
        fontsize=13, fontweight='bold', y=0.99
    )

    # Two side-by-side panels (one per class)
    outer = GridSpec(1, 2, figure=fig, wspace=0.08,
                     top=0.91, bottom=0.06, left=0.07, right=0.98)

    for col, cls in enumerate(classes):
        x_orig, x_s, x_u = decompositions[cls]
        plot_class_panel(fig, outer, col, cls, x_orig, x_s, x_u, N_COLS)

    # ── Annotation box explaining what S and U mean ────────────────────────────
    fig.text(0.5, 0.01,
             'S-space (blue row): captures structural & contextual features shared across classes  '
             '|  '
             'U-space (purple row): captures class-specific discriminative features\n'
             'DKF unlearning erases ONLY the U-space for the forget class  ->  '
             'S-space features (sky, wings shape) are preserved in retain classes',
             ha='center', va='bottom', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.9))

    out = os.path.join(RESULTS_DIR, 'su_decomposition.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {out}")


if __name__ == '__main__':
    main()
