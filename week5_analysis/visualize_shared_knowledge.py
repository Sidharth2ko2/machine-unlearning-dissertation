"""
Shared Knowledge Preservation Visualization.

Shows HOW the β-VAE + DKF approach preserves shared knowledge:

1. COUNTERFACTUAL GRID
   Row 1: original airplane images (X_f)
   Row 2: counterfactuals  X_c = Decoder(S_airplane, U_bird)
           → airplane's shared features (wings, sky) remapped into bird context
   Row 3: actual bird images (X_r)
   Interpretation: the structural similarity between row 1 and row 2 is the
   shared knowledge that DKF preserves — it never erases these features.

2. COSINE SIMILARITY BARS
   For every retain class: compute avg cosine_sim(S_airplane, S_class)
   High similarity → classes share latent structure in S-space.
   This is what DKF protects during unlearning — only U is targeted.

3. RECONSTRUCTION QUALITY
   Reconstruct retain-class images through the VAE (encode S+U, decode back).
   If reconstruction is good, the VAE has faithfully captured retain knowledge.

Run from week5_analysis/:
  uv run python visualize_shared_knowledge.py

Output: week5_analysis/results/shared_knowledge_viz.png
"""
import os
import sys

# ── Point imports to week3-4 shared modules ───────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_W34  = os.path.normpath(os.path.join(_HERE, '..', 'week3-4'))
sys.path.insert(0, _W34)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from torch.utils.data import DataLoader, Subset
import torchvision

from beta_vae import BetaVAE
from config import (
    LATENT_DIM, NUM_CLASSES, BETA, CLASS_NAMES, FORGET_CLASS, VAE_PRETRAIN_EPOCHS
)
from config import get_device
from data_utils import get_transforms

# ── Path overrides ─────────────────────────────────────────────────────────────
DATA_DIR       = os.path.normpath(os.path.join(_W34, '..', 'week1_baseline', 'data'))
CHECKPOINT_DIR = os.path.join(_W34, 'checkpoints')
RESULTS_DIR    = os.path.join(_HERE, 'results')

# ── CIFAR-10 un-normalise for display ─────────────────────────────────────────
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)


def to_display(tensor):
    """Undo CIFAR-10 normalisation and clamp to [0,1] for imshow."""
    img = tensor.cpu() * STD + MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_vae(device):
    path = os.path.join(CHECKPOINT_DIR,
                        f'vae_pretrained_e{VAE_PRETRAIN_EPOCHS}_b{int(BETA)}.pth')
    if not os.path.exists(path):
        sys.exit(
            f"\n[ERROR] VAE checkpoint not found: {path}\n"
            "Run:  cd ../week3-4 && uv run python run_experiments.py\n"
        )
    vae = BetaVAE(latent_dim_s=LATENT_DIM, latent_dim_u=LATENT_DIM,
                  num_classes=NUM_CLASSES, beta=BETA).to(device)
    vae.load_state_dict(torch.load(path, map_location=device))
    vae.eval()
    return vae


def class_samples(class_idx, n=6, use_train=False):
    _, test_tf = get_transforms()
    ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=use_train, download=False, transform=test_tf)
    idx = np.where(np.array(ds.targets) == class_idx)[0][:n]
    loader = DataLoader(Subset(ds, idx), batch_size=n, shuffle=False)
    x, y = next(iter(loader))
    return x, y


# ── Section 1: Counterfactual Grid ────────────────────────────────────────────

def make_counterfactual_grid(vae, device, n_cols=6):
    """
    Generate X_c = Decoder(S_airplane, U_bird) for n_cols airplane images.
    Returns: (airplane images, counterfactuals, bird images) all on CPU.
    """
    x_airplane, _ = class_samples(FORGET_CLASS, n=n_cols, use_train=True)
    x_bird,     _ = class_samples(2,            n=n_cols, use_train=True)

    x_airplane = x_airplane.to(device)
    x_bird     = x_bird.to(device)

    with torch.no_grad():
        s_f, _, _ = vae.encode_shared(x_airplane)
        u_r, _, _ = vae.encode_unique(x_bird)
        x_cf = vae.decoder(s_f, u_r)

    return x_airplane.cpu(), x_cf.cpu(), x_bird.cpu()


# ── Section 2: Cosine Similarity in S and U spaces ────────────────────────────

@torch.no_grad()
def compute_space_similarity(vae, device, space='s', n_per_class=200):
    """
    Avg cosine similarity between airplane mean and each retain class mean
    in either S-space (space='s') or U-space (space='u').
    """
    _, test_tf = get_transforms()
    ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=False, transform=test_tf)
    targets = np.array(ds.targets)
    encoder = vae.encoder_s if space == 's' else vae.encoder_u

    # Airplane mean embedding
    air_idx    = np.where(targets == FORGET_CLASS)[0][:n_per_class]
    air_loader = DataLoader(Subset(ds, air_idx), batch_size=64, shuffle=False)
    z_air = torch.cat([encoder(x.to(device))[0].cpu() for x, _ in air_loader])
    z_air_mean = z_air.mean(0, keepdim=True)

    sims = {}
    for cls in range(NUM_CLASSES):
        if cls == FORGET_CLASS:
            continue
        idx    = np.where(targets == cls)[0][:n_per_class]
        loader = DataLoader(Subset(ds, idx), batch_size=64, shuffle=False)
        z_cls  = torch.cat([encoder(x.to(device))[0].cpu() for x, _ in loader])
        z_cls_mean = z_cls.mean(0, keepdim=True)
        sims[cls] = F.cosine_similarity(z_air_mean, z_cls_mean, dim=1).item()
    return sims


# ── Section 3: VAE Reconstruction of Retain Class ─────────────────────────────

@torch.no_grad()
def make_reconstruction_row(vae, retain_class, device, n=6):
    """Encode + decode retain-class images to verify VAE preserves their features."""
    x, _ = class_samples(retain_class, n=n, use_train=False)
    x = x.to(device)
    s, _, _ = vae.encode_shared(x)
    u, _, _ = vae.encode_unique(x)
    return x.cpu(), vae.decoder(s, u).cpu()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"[Device] {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    vae = load_vae(device)
    print("[VAE] Loaded\n")

    N_COLS = 6

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'How Shared Knowledge is Preserved in DKF\n'
        'CIFAR-10  |  Forget class: Airplane  |  β-VAE Disentanglement',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3,
                  top=0.92, bottom=0.04, left=0.06, right=0.97)

    # ── Panel A: Counterfactual Grid ──────────────────────────────────────────
    print("[Panel A] Generating counterfactuals Decoder(S_airplane, U_bird) …")
    x_air, x_cf, x_bird = make_counterfactual_grid(vae, device, N_COLS)

    gs_top = GridSpecFromSubplotSpec(
        3, N_COLS, subplot_spec=gs[0, :], hspace=0.08, wspace=0.05)

    row_colors = ['#e74c3c', '#9b59b6', '#27ae60']
    row_labels = [
        'Airplane\n(original X_f)',
        'Counterfactual X_c\n=Decoder(S_airplane, U_bird)',
        'Bird\n(original X_r)',
    ]
    for row_i, (imgs, label, color) in enumerate(
            zip([x_air, x_cf, x_bird], row_labels, row_colors)):
        for col_i in range(N_COLS):
            ax = fig.add_subplot(gs_top[row_i, col_i])
            ax.imshow(to_display(imgs[col_i]))
            ax.axis('off')
            if col_i == 0:
                ax.set_ylabel(label, fontsize=8, color=color,
                              fontweight='bold', rotation=90, labelpad=4)
            if row_i == 0 and col_i == N_COLS // 2:
                ax.set_title(
                    'Panel A — Counterfactual Images\n'
                    'Airplane shared features (wings, shape) remapped into bird context\n'
                    '→ shared knowledge travels through X_c without being erased',
                    fontsize=9, fontweight='bold', color='#2c3e50', pad=6)

    fig.add_artist(plt.Line2D([0.04, 0.97], [0.50, 0.50],
                              transform=fig.transFigure,
                              color='#bdc3c7', linewidth=1))

    # ── Panel B: Cosine Similarity Bars ───────────────────────────────────────
    print("[Panel B] Computing cosine similarities in S and U spaces …")
    s_sims = compute_space_similarity(vae, device, space='s')
    u_sims = compute_space_similarity(vae, device, space='u')

    retain_classes = [c for c in range(NUM_CLASSES) if c != FORGET_CLASS]
    retain_names   = [CLASS_NAMES[c] for c in retain_classes]
    s_vals = [s_sims[c] for c in retain_classes]
    u_vals = [u_sims[c] for c in retain_classes]

    ax_sim = fig.add_subplot(gs[1, 0])
    x_pos  = np.arange(len(retain_classes))
    width  = 0.38
    bars_s = ax_sim.bar(x_pos - width/2, s_vals, width,
                        color='#3498db', alpha=0.85, label='S (shared) space')
    bars_u = ax_sim.bar(x_pos + width/2, u_vals, width,
                        color='#e74c3c', alpha=0.85, label='U (unique) space')
    ax_sim.set_xticks(x_pos)
    ax_sim.set_xticklabels(retain_names, rotation=30, ha='right', fontsize=8)
    ax_sim.set_ylabel('Cosine Similarity with Airplane', fontsize=9)
    ax_sim.set_ylim(-0.1, 1.05)
    ax_sim.axhline(0, color='#7f8c8d', linewidth=0.8)
    ax_sim.legend(fontsize=8)
    ax_sim.set_title(
        'Panel B — Cosine Similarity: Airplane vs Retain Classes\n'
        'S-space HIGH → shared structure preserved  |  U-space lower → unique features differ',
        fontsize=9, fontweight='bold', color='#2c3e50')
    ax_sim.grid(axis='y', alpha=0.3, linewidth=0.5)

    for bar in bars_s:
        h = bar.get_height()
        ax_sim.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=6.5)
    for bar in bars_u:
        h    = bar.get_height()
        ypos = h + 0.01 if h >= 0 else h - 0.03
        ax_sim.text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=6.5)

    # ── Panel C: VAE Reconstruction of Retain Class ───────────────────────────
    print("[Panel C] VAE reconstruction of bird images …")
    x_orig, x_recon = make_reconstruction_row(vae, retain_class=2, device=device, n=N_COLS)

    gs_recon = GridSpecFromSubplotSpec(
        2, N_COLS, subplot_spec=gs[1, 1], hspace=0.1, wspace=0.05)
    recon_rows   = [x_orig, x_recon]
    recon_labels = ['Bird (original)', 'Bird (VAE recon)']
    recon_colors = ['#27ae60', '#8e44ad']
    for row_i, (imgs, label, color) in enumerate(
            zip(recon_rows, recon_labels, recon_colors)):
        for col_i in range(N_COLS):
            ax = fig.add_subplot(gs_recon[row_i, col_i])
            ax.imshow(to_display(imgs[col_i]))
            ax.axis('off')
            if col_i == 0:
                ax.set_ylabel(label, fontsize=8, color=color,
                              fontweight='bold', rotation=90, labelpad=4)
            if row_i == 0 and col_i == N_COLS // 2:
                ax.set_title(
                    'Panel C — VAE Reconstruction of Retain Class (bird)\n'
                    'Good reconstruction → shared features captured faithfully\n'
                    '→ DKF will NOT destroy these when forgetting airplane',
                    fontsize=9, fontweight='bold', color='#2c3e50', pad=6)

    out = os.path.join(RESULTS_DIR, 'shared_knowledge_viz.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {out}")

    # Numeric summary
    print("\n── Cosine Similarity Summary ─────────────────────────────────")
    print(f"{'Class':<12}  {'S-space':>8}  {'U-space':>8}  {'Difference':>10}")
    print("-" * 46)
    for cls in retain_classes:
        s = s_sims[cls]
        u = u_sims[cls]
        print(f"{CLASS_NAMES[cls]:<12}  {s:>8.4f}  {u:>8.4f}  {s-u:>+10.4f}")
    print("─" * 46)
    print("Higher S-sim and lower U-sim → better disentanglement\n")


if __name__ == '__main__':
    main()
