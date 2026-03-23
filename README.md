# Machine Unlearning — Dissertation Research Log

**Title:** Disentangled Knowledge Forgetting in Machine Unlearning
**Student:** Sidharth M (2024H1120009U)
**Supervisor:** Prof. Shivang
**Institution:** BITS Pilani

---

## What is Machine Unlearning?

Machine Unlearning is the process of removing the influence of specific training data from an already-trained model — without retraining the model from scratch. This is motivated by privacy regulations (like GDPR's "right to be forgotten") where a user can request their data to be deleted not just from the database, but from any model trained on it.

The challenge is that modern deep learning models like ResNet implicitly memorize training data in their weights. Simply deleting the data from the dataset does not remove its influence from the model. Retraining from scratch is the perfect solution but is computationally expensive at scale.

The base paper — **Disentangled Knowledge Forgetting in Machine Unlearning** — proposes a method called DKF that addresses two failure modes in existing unlearning methods:
- **Overly unlearning (low fidelity):** The model forgets too aggressively and also loses knowledge of classes it should retain.
- **Incomplete unlearning (low effectiveness):** The model does not fully forget the target data — traces of it remain detectable.

---

## Experimental Setup (Common Across All Weeks)

| Component | Choice | Reason |
|-----------|--------|--------|
| Dataset | CIFAR-10 | Standard benchmark, used directly in the paper (Table 1), fast to train |
| Model | ResNet-18 | Same as paper's CIFAR-10 experiments |
| Unlearning scenario | Single-class forgetting | Simplest and most interpretable setting |
| Forget class | airplane (class 0) | 10% of training data — meaningful but not too large |
| Forget set D_f | 5,000 airplane images | All training samples of class 0 |
| Retain set D_r | 45,000 images (9 classes) | All training samples except class 0 |

---

## Week 1 — Baseline Model Training

### Goal
Train a ResNet-18 model on the full CIFAR-10 training set (50,000 images across 10 classes). This model is the **pre-unlearning model** — it has seen all the data including the forget set. All unlearning methods will start from this model.

### Dataset
CIFAR-10 consists of 60,000 colour images (32×32 pixels) across 10 classes:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

- 50,000 training images (5,000 per class)
- 10,000 test images (1,000 per class)
- Publicly available, auto-downloaded via torchvision

### Data Preprocessing
Standard augmentation was applied to the training set to improve generalization:
- **RandomCrop** (32×32 with padding=4) — simulates slight positional variation
- **RandomHorizontalFlip** — doubles effective training diversity
- **Normalize** with CIFAR-10 mean `(0.4914, 0.4822, 0.4465)` and std `(0.2023, 0.1994, 0.2010)`

Test set used only normalization (no augmentation).

### Model Architecture: ResNet-18
ResNet-18 is a convolutional neural network with residual (skip) connections that allow gradients to flow through deep networks without vanishing. It has 18 layers and ~11 million parameters. The final fully-connected layer was modified from 1000 outputs (ImageNet) to 10 outputs (CIFAR-10 classes).

### Training Configuration
| Hyperparameter | Value | Reason |
|----------------|-------|--------|
| Optimizer | SGD | Standard for CIFAR training, matches paper |
| Learning rate | 0.1 | Standard starting LR for SGD on CIFAR |
| Momentum | 0.9 | Helps SGD escape local minima |
| Weight decay | 5e-4 | L2 regularization to prevent overfitting |
| LR Scheduler | CosineAnnealingLR | Smoothly decays LR from 0.1 to ~0 over training |
| Batch size | 128 | Standard for CIFAR-10 |
| Epochs | 100 | Sufficient for convergence |
| Device | Apple MPS | Apple Silicon GPU acceleration |

### Attempt 1 — Training for 50 Epochs

Initially planned to train for 100 epochs but started with 50 to get quick results. Achieved **86.76% test accuracy**.

Then attempted to resume training for another 50 epochs (51–100) to reach the full 100 epoch target.

**Problem discovered:** The resume introduced a broken learning rate schedule. Here is what went wrong:

- In the first run (epochs 1–50), the scheduler used `T_max=50`, so the LR decayed smoothly from 0.1 down to ~0.0001 by epoch 50. The model was well-converged.
- When resuming, a new scheduler was created with `T_max=100` and fast-forwarded 49 steps. At step 49 of a 100-step cosine schedule, the LR is approximately **0.05** — a 500× jump from where training had ended.
- Additionally, the optimizer's momentum buffer (gradient history) was lost since it was not saved in the checkpoint.
- This caused the model to temporarily "un-learn" what it had learned, visible as a sharp accuracy drop at epoch 60.

**Results from the broken run (not used):**

| Epoch | Train Acc | Test Acc | Note |
|-------|-----------|----------|------|
| 10 | 71.2% | 72.0% | |
| 20 | 77.4% | 74.4% | |
| 30 | 82.3% | 80.6% | |
| 40 | 89.2% | 84.5% | |
| 50 | 94.4% | 86.8% | End of first run |
| 60 | 84.0% | 79.9% | **LR spike on resume — accuracy crashed** |
| 70 | 87.7% | 83.8% | Recovering |
| 80 | 91.7% | 85.1% | |
| 90 | 95.9% | 86.7% | |
| 100 | 97.5% | 87.3% | |

Best accuracy: **87.40%** — but the broken schedule meant this was not a valid training curve for a paper.

### Attempt 2 — Clean Training 0→100 Epochs

Discarded the old checkpoint and retrained from scratch with a single clean run of 100 epochs. The `--resume` flag was not used. The CosineAnnealingLR ran smoothly from `LR=0.1` at epoch 1 down to `LR≈0` at epoch 100 with no interruption. Optimizer momentum was maintained throughout.

**Results (used for all subsequent experiments):**

| Epoch | Train Acc | Test Acc |
|-------|-----------|----------|
| 10 | 71.3% | 69.1% |
| 20 | 75.7% | 70.8% |
| 30 | 78.0% | 73.6% |
| 40 | 80.1% | 78.5% |
| 50 | 82.2% | 75.6% |
| 60 | 84.8% | 82.5% |
| 70 | 87.9% | 82.5% |
| 80 | 92.0% | 85.6% |
| 90 | 96.1% | 87.6% |
| **100** | **97.7%** | **88.2%** |

**Best Test Accuracy: 88.27%** ✅

The improvement over the broken run (+0.87%) came purely from fixing the training procedure — same architecture, same hyperparameters, same data.

### Forget / Retain Split

After training, the dataset was split into the unlearning experiment sets:

```
Training set (50,000)
├── Forget set D_f  →  5,000 images  (class 0: airplane)
└── Retain set D_r  → 45,000 images  (classes 1–9: all others)

Test set (10,000) → used as validation throughout
```

This split is handled automatically in `data_utils.py` and confirmed in training output:
```
Forget class : airplane (class index 0)
Forget size  : 5,000
Retain size  : 45,000
```

### Week 1 Summary

| Item | Status | Detail |
|------|--------|--------|
| CIFAR-10 downloaded | ✅ | Auto-downloaded via torchvision |
| ResNet-18 trained | ✅ | 88.27% test accuracy, 100 epochs |
| Forget set D_f defined | ✅ | 5,000 airplane images |
| Retain set D_r defined | ✅ | 45,000 images, 9 classes |
| Checkpoint saved | ✅ | `checkpoints/original_model.pth` |

---

## Week 2 — Baseline Unlearning Methods (In Progress)

### Goal
Implement and evaluate three baseline unlearning methods from the paper and compare them using four standard metrics. This produces a results table equivalent to Table 1 in the paper before the paper's proposed DKF method is implemented.

### Methods to Implement

**1. Retrain (Gold Standard)**
Train a completely fresh ResNet-18 from scratch, but only on the retain set D_r (45,000 images, no airplanes). This is the perfect unlearning reference — the model never saw airplane images, so it has no airplane knowledge by definition. Every other method is measured against how close it gets to this.

**2. Fine-tune**
Take the original trained model (88.27%) and continue training it only on D_r. Without airplane images in the training loop, the model gradually overwrites its airplane knowledge. Fast and simple, but may not fully erase the forget class since the original weights still encode airplane features.

**3. Negative Gradient (NegGrad)**
Simultaneously apply gradient ascent on D_f (maximize loss on airplanes — actively degrade airplane classification) while applying normal gradient descent on D_r (preserve knowledge of the other 9 classes). More aggressive than fine-tuning but risks destabilizing the model if not balanced carefully.

### Evaluation Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Acc_Dr** | Accuracy on retain set — did the model keep its knowledge of the 9 retain classes? | High ↑ |
| **Acc_Df** | Accuracy on forget set — can the model still classify airplanes correctly? | Low ↓ (near 0%) |
| **Acc_val** | Overall test set accuracy | High ↑ |
| **MIA** | Membership Inference Attack — can an attacker detect that airplane images were ever in training? | Low ↓ (near 50% = random) |

---

## Project Structure

```
experiments/
├── config.py            # All hyperparameters and device detection
├── data_utils.py        # CIFAR-10 loading, forget/retain split
├── train_original.py    # ResNet-18 training script
├── checkpoints/         # Saved model weights (not tracked in git)
│   └── original_model.pth
├── data/                # CIFAR-10 dataset (not tracked in git)
├── results/             # Evaluation outputs (not tracked in git)
└── pyproject.toml       # uv dependency management
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Sidharth2ko2/machine-unlearning-dissertation.git
cd machine-unlearning-dissertation/experiments

# Set up environment with uv
uv venv && uv sync

# Train baseline model (clean 100 epoch run)
uv run python train_original.py --epochs 100
```

### Dependencies
- Python 3.12
- PyTorch 2.10.0 + torchvision 0.25.0
- scikit-learn, numpy, matplotlib, tqdm
- Platform: Apple MPS (Metal Performance Shaders) / CUDA / CPU auto-detected

---

## References

- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009
- Base paper: Disentangled Knowledge Forgetting in Machine Unlearning (Anonymous submission)
- Chundawat et al., "Zero-Shot Machine Unlearning", 2023
- Golatkar et al., "Eternal Sunshine of the Spotless Net", CVPR 2020
