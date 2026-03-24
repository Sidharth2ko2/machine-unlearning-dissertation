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

## Week 2 — Baseline Unlearning Methods

### Goal
Implement and evaluate three baseline unlearning methods from the paper and compare them using four standard metrics. This produces a results table equivalent to Table 1 in the paper, establishing a performance baseline before implementing the paper's proposed DKF method.

### Repository Restructure
At the start of Week 2, the repo was reorganised from a flat structure into a week-by-week folder layout. Week 1 files were moved into `week1_baseline/` and `week2_unlearning/` was created as a clean folder. The `data/` and `checkpoints/` directories (gitignored) were also moved into `week1_baseline/`. Week 2's config points to Week 1's data to avoid re-downloading CIFAR-10.

### Method 1 — Retrain (Gold Standard)
**What it does:** Discard the original model entirely. Train a completely new ResNet-18 from scratch, but only on the retain set D_r (45,000 images, no airplanes whatsoever).

**Why it is the gold standard:** A model that never saw airplane images cannot have any airplane knowledge. This is the theoretically perfect answer to machine unlearning. All other methods are measured by how closely they match this.

**The problem:** Takes as long as original training (~37 minutes). At industrial scale, this is infeasible — if 10,000 users request deletion in a day, you cannot retrain 10,000 times. This motivates approximate unlearning methods.

**Implementation:** Fresh ResNet-18, SGD with cosine LR decay, trained 100 epochs on D_r only.

---

### Method 2 — Fine-tune
**What it does:** Take the original trained model (88.27%) and continue training it on D_r only. No airplane images are ever shown during fine-tuning, so the model gradually overwrites its airplane representations.

**Why it works partially:** The optimizer is pushing weights toward a solution that fits D_r. Over time, features that were specifically tuned for airplanes get repurposed for other classes.

**Why it may be incomplete:** The original weights already contain airplane knowledge baked in. Fine-tuning does not explicitly erase those features — it just stops reinforcing them. Some residual airplane knowledge often remains detectable via MIA.

**Implementation:** Same model weights as baseline, SGD at lower LR (0.01) for 10 epochs on D_r.

---

### Method 3 — Negative Gradient (NegGrad)
**What it does:** Simultaneously perform two opposing updates in each training step:
- **Gradient ascent on D_f** → Maximise the loss on airplane images (deliberately make the model worse at airplanes)
- **Gradient descent on D_r** → Minimise the loss on retain images (keep the model good at the other 9 classes)

**Combined loss:** `L = L_retain - α × L_forget`

where `α = 0.5` controls the balance. A higher α means more aggressive forgetting but risks destabilising the overall model. Gradient clipping (`max_norm=1.0`) is applied to prevent the ascent step from causing divergence.

**Why it is more targeted:** Unlike fine-tuning, which passively forgets, NegGrad actively pushes the model away from airplane predictions. This often results in lower Acc_Df and lower MIA.

**The tradeoff:** Too high an α can cause the retain accuracy to drop as well (collateral damage), because shared features between airplanes and other classes may be disrupted.

**Implementation:** Original model weights, SGD at LR 0.01, 10 epochs, cycling through D_f in sync with D_r batches.

---

### Evaluation Metrics

| Metric | What it measures | Target | Why it matters |
|--------|-----------------|--------|----------------|
| **Acc_Dr** | Accuracy on retain set | High ↑ | Model must not forget the 9 classes it should keep |
| **Acc_Df** | Accuracy on forget set | Low ↓ (~0%) | Direct measure of whether forgetting worked |
| **Acc_val** | Overall test accuracy | High ↑ | Ensures the model is still useful generally |
| **MIA** | Membership Inference Attack accuracy | Low ↓ (~50%) | Privacy metric — can an attacker prove airplane data was used? |
| **Avg.Gap** | Mean absolute gap from Retrain across all 4 metrics | Low ↓ | Summary score — how close are we to perfect unlearning? |

### Membership Inference Attack — Explained
An MIA tries to answer: *"Was this sample in the training data?"*

The loss-based approach used here:
1. Compute per-sample cross-entropy loss on D_f (these were in training)
2. Compute per-sample cross-entropy loss on test set (these were not in training)
3. If the model was trained on D_f → its loss on D_f will be very low (it memorised them)
4. Train a logistic regression classifier: can it separate D_f from test samples using loss alone?
5. Report accuracy of this classifier

**Interpretation:**
- MIA = 100% → model perfectly remembers all forget samples (unlearning completely failed)
- MIA = 50% → model treats forget samples identically to unseen test data (perfect unlearning)
- MIA for Retrain → should be near 50% since D_f was never used in training

### Results (To be filled after running experiments)

| Method | Acc_Dr (↑) | Acc_Df (↓) | Acc_val (↑) | MIA (↓) | Avg. Gap |
|--------|-----------|-----------|------------|--------|---------|
| Original | — | — | — | — | — |
| Retrain | — | — | — | — | 0.00 |
| Fine-tune | — | — | — | — | — |
| NegGrad | — | — | — | — | — |

---

## Project Structure

Each week is a self-contained folder. Shared environment (`pyproject.toml`, `.venv`) lives at the `experiments/` root.

```
experiments/
├── week1_baseline/
│   ├── config.py            # Hyperparameters and device detection
│   ├── data_utils.py        # CIFAR-10 loading, forget/retain split
│   ├── train_original.py    # ResNet-18 training script (--resume supported)
│   ├── checkpoints/         # Saved weights — not tracked in git
│   │   └── original_model.pth
│   └── data/                # CIFAR-10 dataset — not tracked in git
│
├── week2_unlearning/
│   ├── config.py            # Inherits Week 1 settings, points to Week 1 data/model
│   ├── data_utils.py        # Same loader, download=False (reuses Week 1 data)
│   ├── unlearn.py           # Retrain, Fine-tune, NegGrad implementations
│   ├── evaluate.py          # Acc_Dr, Acc_Df, Acc_val, MIA metrics
│   ├── run_experiments.py   # Main runner — produces results table
│   ├── checkpoints/         # Unlearned model weights — not tracked in git
│   └── results/             # JSON results output — not tracked in git
│
├── README.md                # This research log
├── pyproject.toml           # Shared uv dependencies
└── uv.lock
```

---

## How to Run

```bash
# Clone and set up
git clone https://github.com/Sidharth2ko2/machine-unlearning-dissertation.git
cd machine-unlearning-dissertation/experiments
uv venv && uv sync

# Week 1 — Train baseline model
cd week1_baseline
uv run python train_original.py --epochs 100

# Week 2 — Run unlearning experiments (requires Week 1 checkpoint)
cd ../week2_unlearning
uv run python run_experiments.py
uv run python run_experiments.py --skip-retrain   # skip the 37-min retrain if already done
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
