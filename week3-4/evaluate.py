"""
Week 2 — Evaluation Metrics for Machine Unlearning

Implements all four metrics from the paper (Table 1):
  1. Acc_Dr  — accuracy on retain set   (should stay HIGH)
  2. Acc_Df  — accuracy on forget set   (should drop LOW)
  3. Acc_val — overall test accuracy    (should stay HIGH)
  4. MIA     — Membership Inference Attack score (should drop LOW)
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ── Core accuracy helper ────────────────────────────────────────────────────────

@torch.no_grad()
def accuracy(model, loader, device):
    """Compute classification accuracy on a given dataloader."""
    model.eval()
    correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs).argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total   += targets.size(0)
    return 100.0 * correct / total


# ── Per-sample loss (needed for MIA) ───────────────────────────────────────────

@torch.no_grad()
def per_sample_loss(model, loader, device):
    """
    Compute cross-entropy loss for every individual sample.
    Lower loss = model is more confident = likely saw this sample during training.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss = criterion(model(inputs), targets)
        losses.extend(loss.cpu().numpy())
    return np.array(losses)


# ── Membership Inference Attack ─────────────────────────────────────────────────

def membership_inference_attack(model, forget_loader, forget_class_test_loader, device):
    """
    Loss-based Membership Inference Attack (MIA) — class-balanced version.

    Compares D_f (forget-class training images) against forget-class TEST images.
    Both groups are the same class (airplane), so the only difference is
    whether the model was trained on them or not.

    If unlearning worked:
      - D_f loss ≈ forget-class test loss (model treats both as unseen)
      - Attacker can't distinguish → MIA accuracy ≈ 50% → GOOD

    If unlearning failed:
      - D_f loss still very low (memorised), test loss higher
      - Attacker easily separates them → MIA accuracy HIGH → BAD

    Using full mixed test set (old approach) was wrong for Retrain/NegGrad because
    the model has high loss on ALL airplane images — confounding membership with class.
    Class-balanced MIA isolates the true privacy signal.

    Returns MIA accuracy (%) — lower is better, 50% = perfect unlearning.
    """
    forget_losses = per_sample_loss(model, forget_loader,            device)
    test_losses   = per_sample_loss(model, forget_class_test_loader, device)

    n = min(len(forget_losses), len(test_losses))
    forget_losses = forget_losses[:n]
    test_losses   = test_losses[:n]

    X = np.concatenate([forget_losses, test_losses]).reshape(-1, 1)
    y = np.concatenate([np.ones(n), np.zeros(n)])   # 1=member (D_f), 0=non-member (test)

    clf = LogisticRegression()
    clf.fit(X, y)
    mia_acc = accuracy_score(y, clf.predict(X)) * 100
    return mia_acc


# ── Full evaluation for one model ──────────────────────────────────────────────

def evaluate_model(model, loaders, device):
    """
    Run all four metrics on a model and return as a dict.
    loaders must have keys: 'retain', 'forget', 'test', 'forget_class_test'
    """
    acc_dr  = accuracy(model, loaders['retain'],           device)
    acc_df  = accuracy(model, loaders['forget'],           device)
    acc_val = accuracy(model, loaders['test'],             device)
    mia     = membership_inference_attack(model, loaders['forget'],
                                          loaders['forget_class_test'], device)
    return {'Acc_Dr': acc_dr, 'Acc_Df': acc_df, 'Acc_val': acc_val, 'MIA': mia}


# ── Results table printer ──────────────────────────────────────────────────────

def print_results_table(results: dict):
    """
    Print a comparison table for all methods, matching paper's Table 1 format.
    results = { 'MethodName': {'Acc_Dr': x, 'Acc_Df': x, 'Acc_val': x, 'MIA': x} }
    """
    # Compute Avg Gap vs Retrain
    retrain = results.get('Retrain', None)

    header = f"{'Method':<18} {'Acc_Dr(↑)':>10} {'Acc_Df(↓)':>10} {'Acc_val(↑)':>10} {'MIA(↓)':>10} {'Avg.Gap':>10}"
    print("\n" + "=" * 72)
    print("  Machine Unlearning Results — CIFAR-10 (ResNet-18)")
    print("  Forget class: airplane (class 0)")
    print("=" * 72)
    print(header)
    print("-" * 72)

    for method, m in results.items():
        if retrain and method != 'Retrain':
            gap = np.mean([
                abs(m['Acc_Dr']  - retrain['Acc_Dr']),
                abs(m['Acc_Df']  - retrain['Acc_Df']),
                abs(m['Acc_val'] - retrain['Acc_val']),
                abs(m['MIA']     - retrain['MIA']),
            ])
            gap_str = f"{gap:>9.2f}%"
        else:
            gap_str = f"{'—':>10}"

        print(f"{method:<18} {m['Acc_Dr']:>9.2f}% {m['Acc_Df']:>9.2f}% "
              f"{m['Acc_val']:>9.2f}% {m['MIA']:>9.2f}% {gap_str}")

    print("=" * 72)
    print("\nInterpretation:")
    print("  Acc_Dr  — accuracy on retain set    → higher is better")
    print("  Acc_Df  — accuracy on forget set    → lower is better (near 0%)")
    print("  Acc_val — overall test accuracy     → higher is better")
    print("  MIA     — membership inference acc  → lower is better (near 50%)")
    print("  Avg.Gap — mean gap from Retrain     → lower is better\n")
