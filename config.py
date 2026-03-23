"""
Centralized configuration for the Machine Unlearning experiments.
Paper: Disentangled Knowledge Forgetting in Machine Unlearning
Dataset: CIFAR-10 | Model: ResNet-18 | Setting: Single-class forgetting
"""
import os
import torch

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET         = 'cifar10'
NUM_CLASSES     = 10
FORGET_CLASS    = 0          # 0 = 'airplane' (single-class forgetting scenario)
CLASS_NAMES     = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE      = 128
LR_ORIGINAL     = 0.1        # SGD lr for original training
LR_UNLEARN      = 0.01       # lr for unlearning fine-tuning
WEIGHT_DECAY    = 5e-4
MOMENTUM        = 0.9
EPOCHS_ORIGINAL = 100        # original model training epochs
EPOCHS_UNLEARN  = 10         # epochs for fine-tune / neg-grad unlearning
EPOCHS_RETRAIN  = 50         # epochs for retrain-from-scratch (gold standard)
NEG_GRAD_ALPHA  = 0.5        # balance between forget-ascent and retain-descent

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = './data'
CHECKPOINT_DIR  = './checkpoints'
RESULTS_DIR     = './results'


def get_device():
    """Auto-detect best available device: CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except AttributeError:
        pass
    return torch.device('cpu')


def setup_dirs():
    for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
