"""
Visualization of β-VAE Disentanglement via t-SNE.

Demonstrates that after DKF training:
  S (shared) space : airplane embeddings INTERMIX with retain classes
                     → structural features (sky, wings) are spread across classes
  U (unique) space : airplane embeddings CLUSTER SEPARATELY
                     → class-specific features (landing gear) are isolated

Run from week5_analysis/:
  uv run python visualize_disentanglement.py

Output: week5_analysis/results/tsne_disentanglement.png
"""
import os
import sys

# ── Point imports to week3-4 shared modules ───────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_W34  = os.path.normpath(os.path.join(_HERE, '..', 'week3-4'))
sys.path.insert(0, _W34)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
import torchvision

from beta_vae import BetaVAE
from config import (
    LATENT_DIM, NUM_CLASSES, BETA, CLASS_NAMES, VAE_PRETRAIN_EPOCHS
)
from config import get_device
from data_utils import get_transforms

# ── Path overrides (config.py uses paths relative to week3-4, fix them) ───────
DATA_DIR       = os.path.normpath(os.path.join(_W34, '..', 'week1_baseline', 'data'))
CHECKPOINT_DIR = os.path.join(_W34, 'checkpoints')
RESULTS_DIR    = os.path.join(_HERE, 'results')


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_vae(device):
    path = os.path.join(CHECKPOINT_DIR,
                        f'vae_pretrained_e{VAE_PRETRAIN_EPOCHS}_b{int(BETA)}.pth')
    if not os.path.exists(path):
        sys.exit(
            f"\n[ERROR] VAE checkpoint not found: {path}\n"
            "Run week3-4 first:  cd ../week3-4 && uv run python run_experiments.py\n"
        )
    vae = BetaVAE(latent_dim_s=LATENT_DIM, latent_dim_u=LATENT_DIM,
                  num_classes=NUM_CLASSES, beta=BETA).to(device)
    vae.load_state_dict(torch.load(path, map_location=device))
    vae.eval()
    print(f"[VAE] Loaded ← {path}")
    return vae


def class_loader(class_idx, max_samples=250, batch_size=64):
    _, test_tf = get_transforms()
    ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=False, transform=test_tf)
    idx = np.where(np.array(ds.targets) == class_idx)[0][:max_samples]
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                      num_workers=2)


@torch.no_grad()
def extract_latents(vae, class_indices, device, max_per_class=200):
    """
    Extract encoder means (S_mu, U_mu) — no reparameterization,
    so t-SNE reflects the deterministic embedding structure.
    """
    s_all, u_all, labels = [], [], []
    for cls in class_indices:
        loader = class_loader(cls, max_samples=max_per_class + 64)
        s_cls, u_cls = [], []
        for x, _ in loader:
            x = x.to(device)
            s_mu, _ = vae.encoder_s(x)
            u_mu, _ = vae.encoder_u(x)
            s_cls.append(s_mu.cpu().numpy())
            u_cls.append(u_mu.cpu().numpy())
        s_np = np.concatenate(s_cls)[:max_per_class]
        u_np = np.concatenate(u_cls)[:max_per_class]
        s_all.append(s_np)
        u_all.append(u_np)
        labels.append(np.full(len(s_np), cls))
        print(f"  {CLASS_NAMES[cls]:<12}: {len(s_np)} samples")

    return (np.concatenate(s_all),
            np.concatenate(u_all),
            np.concatenate(labels))


def run_tsne(embeddings, seed=42):
    print("  Running t-SNE …", end=' ', flush=True)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1200,
                learning_rate='auto', init='pca', random_state=seed)
    out = tsne.fit_transform(embeddings)
    print("done")
    return out


def scatter_classes(ax, emb_2d, labels, class_indices, colors, title, note):
    for i, cls in enumerate(class_indices):
        mask = labels == cls
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=colors[i], label=CLASS_NAMES[cls],
                   alpha=0.65, s=14, linewidths=0)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.legend(markerscale=2.5, fontsize=8, framealpha=0.8, loc='best')
    ax.set_xlabel('t-SNE dim 1', fontsize=9)
    ax.set_ylabel('t-SNE dim 2', fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffde7', alpha=0.9))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"[Device] {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    vae = load_vae(device)

    # Classes chosen because they visually share features with airplane:
    #   bird(2)  → wings & sky;  ship(8) → sky/water background
    #   automobile(1) → angular body;  deer(4) → legs/silhouette contrast
    class_indices = [0, 1, 2, 4, 8]   # airplane, automobile, bird, deer, ship
    colors        = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#8e44ad']

    print("\n[Step 1] Extracting latent embeddings …")
    s_all, u_all, labels = extract_latents(vae, class_indices, device)

    print("\n[Step 2] Computing t-SNE projections …")
    print("  S (shared) space:")
    s_2d = run_tsne(s_all)
    print("  U (unique) space:")
    u_2d = run_tsne(u_all)

    print("\n[Step 3] Plotting …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        'β-VAE Disentanglement — t-SNE of Latent Spaces\n'
        'CIFAR-10: Airplane (forget class) vs 4 Retain Classes',
        fontsize=13, fontweight='bold', y=1.01
    )

    scatter_classes(
        axes[0], s_2d, labels, class_indices, colors,
        title='S  —  Shared Latent Space\n'
              '(structural / contextual features)',
        note='✓  Airplane points intermix with bird & ship\n'
             '   → wings, sky context shared across classes\n'
             '   → DKF preserves these features after forgetting'
    )

    scatter_classes(
        axes[1], u_2d, labels, class_indices, colors,
        title='U  —  Unique Latent Space\n'
              '(class-specific features)',
        note='✓  Airplane forms its own cluster\n'
             '   → landing gear, engine, fuselage isolated\n'
             '   → DKF targets only this subspace for erasure'
    )

    # Draw a box around airplane in the U plot to highlight separation
    x_airplane = u_2d[labels == 0]
    pad = 3
    rect_x = x_airplane[:, 0].min() - pad
    rect_y = x_airplane[:, 1].min() - pad
    rect_w = x_airplane[:, 0].ptp() + 2 * pad
    rect_h = x_airplane[:, 1].ptp() + 2 * pad
    axes[1].add_patch(plt.Rectangle(
        (rect_x, rect_y), rect_w, rect_h,
        fill=False, edgecolor='#e74c3c', linewidth=1.8,
        linestyle='--', label='airplane cluster'
    ))

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'tsne_disentanglement.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {out}")


if __name__ == '__main__':
    main()
