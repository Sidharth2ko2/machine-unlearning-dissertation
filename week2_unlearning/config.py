"""
Configuration for Week 2 — Baseline Unlearning Methods.
Inherits dataset/model setup from Week 1. Points to Week 1's data and
original model checkpoint to avoid re-downloading or re-training.
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
# Reuse Week 1's downloaded data — no need to re-download CIFAR-10
DATA_DIR             = '../week1_baseline/data'
# Week 1's trained model — starting point for all unlearning methods
ORIGINAL_MODEL_PATH  = '../week1_baseline/checkpoints/original_model.pth'
# Week 2 outputs
CHECKPOINT_DIR       = './checkpoints'
RESULTS_DIR          = './results'

# ── Training hyperparameters (same as Week 1) ──────────────────────────────────
BATCH_SIZE      = 128
LR_ORIGINAL     = 0.1
MOMENTUM        = 0.9
WEIGHT_DECAY    = 5e-4

# ── Unlearning-specific hyperparameters ────────────────────────────────────────
LR_UNLEARN          = 0.01    # lower LR for fine-tuning on retain set
EPOCHS_UNLEARN      = 10      # epochs for Fine-tune and NegGrad
EPOCHS_RETRAIN      = 100     # epochs for Retrain from scratch (gold standard)
NEGGRAD_ALPHA       = 0.5     # weight balancing forget-ascent vs retain-descent


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
