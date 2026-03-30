"""
Configuration for Week 3 — DKF (Disentangled Knowledge Forgetting).
Inherits dataset/model setup from Week 1 and 2.
"""
import os
import torch

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET         = 'cifar10'
NUM_CLASSES     = 10
FORGET_CLASS    = 0
CLASS_NAMES     = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR            = '../week1_baseline/data'
ORIGINAL_MODEL_PATH = '../week1_baseline/checkpoints/original_model.pth'
WEEK2_RESULTS_PATH  = '../week2_unlearning/results/week2_results.json'
CHECKPOINT_DIR      = './checkpoints'
RESULTS_DIR         = './results'

# ── Standard training (for reference) ─────────────────────────────────────────
BATCH_SIZE      = 128
MOMENTUM        = 0.9
WEIGHT_DECAY    = 5e-4

# ── β-VAE hyperparameters ──────────────────────────────────────────────────────
LATENT_DIM      = 64    # dimensionality of both S (shared) and U (unique) latent spaces
BETA            = 4.0   # β=4 gives sharper counterfactuals than β=6 at short VAE
                        # pretrain (5 epochs). β=6 needs much longer training to
                        # produce clean x_cf — at 5 epochs, β=6 counterfactuals
                        # are blurry → teacher(x_cf) ≈ uniform → destructive L_f.

# ── DKF training hyperparameters ───────────────────────────────────────────────
VAE_PRETRAIN_EPOCHS = 5     # Phase 1: pre-train VAE alone so counterfactuals
                            # are stable before student training begins.
DKF_EPOCHS          = 10   # Phase 2: student training epochs (VAE frozen)
DKF_LR              = 5e-5 # small LR — prevents forget gradient from corrupting backbone
LAMBDA_RETAIN       = 10.0 # strong retain CE anchor (best found across all attempts)
LAMBDA_FORGET       = 0.1  # gentle forgetting signal
LAMBDA_C            = 0.05 # small contrastive with correct direction and detached negatives
TEMPERATURE         = 0.07 # paper's τ


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except AttributeError:
        pass
    return torch.device('cpu')


def setup_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
