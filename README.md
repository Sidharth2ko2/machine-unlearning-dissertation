# Machine Unlearning — Dissertation Implementation

**Paper:** Disentangled Knowledge Forgetting in Machine Unlearning
**Author:** Sidharth M (2024H1120009U)
**Supervisor:** Prof. Shivang

---

## Overview

This repository contains the implementation for my dissertation on **Machine Unlearning** — the task of removing the influence of specific training data from an already-trained model, without retraining from scratch.

The work follows the experimental setup of the base paper and progressively implements baseline unlearning methods before moving to the paper's proposed approach (DKF).

---

## Week 1 — Baseline Model Training ✅

### Objective
Train a ResNet-18 model on CIFAR-10 as the original model (pre-unlearning), and define the forget/retain split for unlearning experiments.

### Dataset: CIFAR-10
| Property | Value |
|----------|-------|
| Total images | 60,000 |
| Training images | 50,000 |
| Test images | 10,000 |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Image size | 32 × 32 RGB |

### Model: ResNet-18
- Standard ResNet-18 architecture
- Final fully-connected layer modified for 10-class output
- Trained with SGD (lr=0.1, momentum=0.9, weight decay=5e-4)
- CosineAnnealingLR scheduler over 100 epochs
- Device: Apple MPS (Metal Performance Shaders)

### Results

| Epoch | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| 10    | 71.2%         | 72.0%         |
| 20    | 77.4%         | 74.4%         |
| 30    | 82.3%         | 80.6%         |
| 40    | 89.2%         | 84.5%         |
| 50    | 94.4%         | 86.8%         |
| 60    | 84.0%         | 79.9%         |
| 70    | 87.7%         | 83.8%         |
| 80    | 91.7%         | 85.1%         |
| 90    | 95.9%         | 86.7%         |
| **100**   | **97.5%**         | **87.3%**         |

**Best Test Accuracy: 87.40%**

### Unlearning Setup

| Component | Details |
|-----------|---------|
| Forget set D_f | 5,000 airplane images (class 0) — 10% of training data |
| Retain set D_r | 45,000 images across remaining 9 classes |
| Forget class | airplane (class index 0) |
| Scenario | Single-class forgetting |

---

## Week 2 — Baseline Unlearning Methods (Upcoming)

Will implement and evaluate three baseline unlearning methods from the paper:

1. **Retrain** — Retrain fresh model only on D_r (gold standard)
2. **Fine-tune** — Continue training original model on D_r only
3. **Negative Gradient (NegGrad)** — Gradient ascent on D_f + gradient descent on D_r

### Evaluation Metrics
| Metric | Description | Goal |
|--------|-------------|------|
| Acc_Dr | Accuracy on retain set | High ↑ |
| Acc_Df | Accuracy on forget set | Low ↓ |
| Acc_val | Test set accuracy | High ↑ |
| MIA | Membership Inference Attack score | Low ↓ |

---

## Project Structure

```
experiments/
├── config.py            # Hyperparameters and device setup
├── data_utils.py        # CIFAR-10 loading, forget/retain split
├── train_original.py    # Baseline ResNet-18 training (with --resume support)
├── checkpoints/         # Saved model weights
│   └── original_model.pth
├── data/                # CIFAR-10 dataset (auto-downloaded)
├── results/             # Evaluation outputs
└── pyproject.toml       # uv project dependencies
```

---

## Setup & Run

```bash
# Clone the repo
git clone https://github.com/Sidharth2ko2/machine-unlearning-dissertation.git
cd machine-unlearning-dissertation/experiments

# Install dependencies with uv
uv venv
uv sync

# Train baseline model (100 epochs)
uv run python train_original.py --epochs 100

# Resume training from checkpoint
uv run python train_original.py --epochs 100 --resume
```

### Dependencies
- Python 3.12
- PyTorch 2.10.0
- torchvision 0.25.0
- scikit-learn, numpy, matplotlib, tqdm

---

## References

- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009
- Base paper: Disentangled Knowledge Forgetting in Machine Unlearning (Anonymous submission)
