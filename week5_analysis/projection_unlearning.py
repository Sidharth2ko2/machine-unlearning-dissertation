"""
Gradient Projection Unlearning (GP-Unlearn).

Core idea:
  Standard NegGrad does gradient ascent on forget data and descent on retain data,
  but the forget gradient can leak into the retain gradient direction → damaging
  retain accuracy.

  GP-Unlearn fixes this by projecting the forget gradient to be orthogonal to
  the retain gradient BEFORE applying the update:

      g_f_proj = g_f - (g_f · g_r / ||g_r||²) × g_r

  The projected forget gradient cannot move parameters in the retain direction,
  so retain knowledge is geometrically protected.

  Update rule:
      θ ← θ  -  lr × g_r          (descent on retain)
               +  lr × α × g_f_proj   (ascent on forget, projected away from retain)

Why this is related to DKF:
  Both methods protect shared knowledge geometrically:
    - DKF       : operates in VAE latent space (S vs U disentanglement)
    - GP-Unlearn: operates in parameter space (retain-orthogonal projection)

  GP-Unlearn is the parameter-space analogue of DKF's latent-space projection.

Run from week5_analysis/:
  uv run python projection_unlearning.py

Output: week5_analysis/results/projection_unlearning_results.json
        week5_analysis/results/projection_comparison_table.txt
"""
import copy
import json
import os
import sys

# ── Point imports to week3-4 shared modules ───────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_W34  = os.path.normpath(os.path.join(_HERE, '..', 'week3-4'))
_W2   = os.path.normpath(os.path.join(_HERE, '..', 'week2_unlearning'))
sys.path.insert(0, _W34)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    CLASS_NAMES, FORGET_CLASS, NUM_CLASSES
)
from config import get_device
from data_utils import get_all_loaders

# ── Path overrides ─────────────────────────────────────────────────────────────
DATA_DIR            = os.path.normpath(os.path.join(_W34, '..', 'week1_baseline', 'data'))
ORIGINAL_MODEL_PATH = os.path.normpath(os.path.join(_W34, '..', 'week1_baseline', 'checkpoints', 'original_model.pth'))
CHECKPOINT_DIR      = os.path.join(_HERE, 'checkpoints')
RESULTS_DIR         = os.path.join(_HERE, 'results')
W2_RESULTS_PATH     = os.path.join(_W2,   'results', 'week2_results.json')
W3_RESULTS_PATH     = os.path.join(_W34,  'results', 'week3_results.json')


# ── Gradient flattening utilities ─────────────────────────────────────────────

def flatten_grads(model: nn.Module) -> torch.Tensor:
    """Concatenate all parameter gradients into one flat vector."""
    device = next(model.parameters()).device
    grads  = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=device))
    return torch.cat(grads)


def set_grads(model: nn.Module, flat_grad: torch.Tensor):
    """Write a flat gradient vector back into each parameter's .grad."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset: offset + n].view(p.shape).clone()
        offset += n


def project_orthogonal(g_f: torch.Tensor, g_r: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """
    Project g_f onto the orthogonal complement of g_r.

    g_f_proj = g_f - (g_f · g_r / ||g_r||²) × g_r

    This removes the component of g_f that aligns with the retain direction,
    guaranteeing the forget update cannot corrupt retain-knowledge parameters.
    """
    dot     = torch.dot(g_f, g_r)
    norm_sq = torch.dot(g_r, g_r) + eps
    return g_f - (dot / norm_sq) * g_r


# ── GP-Unlearn training ───────────────────────────────────────────────────────

def train_gp_unlearn(original_model, forget_loader, retain_loader, device,
                     epochs: int   = 10,
                     lr:     float = 5e-4,
                     alpha:  float = 0.5):
    """
    Gradient Projection Unlearning.

    For each retain-batch / forget-batch pair:
      1. g_r  = ∇ CE(model(x_r), y_r)           [retain gradient]
      2. g_f  = ∇ (−CE(model(x_f), y_f))        [forget ascent gradient]
      3. g_f_proj = g_f − proj(g_f onto g_r)    [orthogonal projection]
      4. update: θ ← θ − lr × (g_r − α × g_f_proj)

    Args:
        original_model : pretrained ResNet-18 (deep-copied, not modified)
        forget_loader  : DataLoader for D_f (airplane images)
        retain_loader  : DataLoader for D_r (retain-class images)
        device         : torch device
        epochs         : unlearning epochs (retain-outer loop)
        lr             : SGD learning rate
        alpha          : weight for projected forget gradient

    Returns:
        Unlearned model
    """
    def _cycle(loader):
        while True:
            yield from loader

    model     = copy.deepcopy(original_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    forget_cycle = _cycle(forget_loader)

    print(f"\n[GP-Unlearn] epochs={epochs}  lr={lr}  α={alpha}")
    print("  Retain-outer loop: all retain batches/epoch, cycling forget loader")

    for epoch in tqdm(range(1, epochs + 1), desc='GP-Unlearn', unit='epoch'):
        model.train()
        total_retain_loss = 0.0
        n_steps           = 0

        for x_r, y_r in retain_loader:
            x_f, y_f = next(forget_cycle)
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)
            b = min(x_r.size(0), x_f.size(0))
            x_r, y_r, x_f, y_f = x_r[:b], y_r[:b], x_f[:b], y_f[:b]

            # Step 1: Retain gradient
            optimizer.zero_grad()
            loss_r = criterion(model(x_r), y_r)
            loss_r.backward()
            g_r = flatten_grads(model)

            # Step 2: Forget ascent gradient (negate loss = ascent)
            optimizer.zero_grad()
            loss_f = -criterion(model(x_f), y_f)
            loss_f.backward()
            g_f = flatten_grads(model)

            # Step 3: Project g_f ⊥ g_r
            g_f_proj = project_orthogonal(g_f, g_r)

            # Step 4: Combined gradient → optimizer applies the update
            g_combined = g_r - alpha * g_f_proj
            set_grads(model, g_combined)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_retain_loss += loss_r.item()
            n_steps           += 1

        avg_loss = total_retain_loss / max(n_steps, 1)
        print(f"  Epoch {epoch:2d}/{epochs}  retain_loss={avg_loss:.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = os.path.join(CHECKPOINT_DIR, 'gp_unlearn_model.pth')
    torch.save(model.state_dict(), ckpt)
    print(f"[GP-Unlearn] Saved → {ckpt}")
    return model


# ── Evaluation (identical logic to week2/week3) ───────────────────────────────

def evaluate(model, loaders, device):
    """Returns dict: Acc_Dr, Acc_Df, Acc_val, MIA."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    def accuracy(loader):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    def per_sample_loss(loader):
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                losses.extend(criterion(model(x), y).cpu().numpy().tolist())
        return np.array(losses)

    acc_dr  = accuracy(loaders['retain'])
    acc_df  = accuracy(loaders['forget'])
    acc_val = accuracy(loaders['test'])

    lf = per_sample_loss(loaders['forget'])
    lt = per_sample_loss(loaders['forget_class_test'])
    n  = min(len(lf), len(lt))
    X_mia = np.concatenate([lf[:n], lt[:n]]).reshape(-1, 1)
    y_mia = np.array([1]*n + [0]*n)
    cv    = min(5, n // 10) if n >= 50 else 2
    mia   = 100.0 * cross_val_score(
        LogisticRegression(max_iter=1000), X_mia, y_mia,
        cv=cv, scoring='accuracy'
    ).mean()

    return {'Acc_Dr': acc_dr, 'Acc_Df': acc_df, 'Acc_val': acc_val, 'MIA': mia}


def avg_gap(m, ref):
    keys = ['Acc_Dr', 'Acc_Df', 'Acc_val', 'MIA']
    return sum(abs(m[k] - ref[k]) for k in keys) / len(keys)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"[Device] {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Patch DATA_DIR into data_utils (it reads from config.py's DATA_DIR)
    import config as _cfg
    _cfg.DATA_DIR = DATA_DIR

    # Load original model
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        sys.exit(f"[ERROR] Original model not found: {ORIGINAL_MODEL_PATH}\n"
                 "Run week1_baseline first.")
    print(f"\n[Loading] original model ← {ORIGINAL_MODEL_PATH}")

    from torchvision.models import resnet18
    original = resnet18(weights=None)
    original.fc = nn.Linear(512, NUM_CLASSES)
    original.load_state_dict(torch.load(ORIGINAL_MODEL_PATH, map_location=device))
    original = original.to(device)
    original.eval()

    loaders = get_all_loaders()

    # ── Train GP-Unlearn ───────────────────────────────────────────────────────
    gp_model = train_gp_unlearn(
        original,
        forget_loader = loaders['forget'],
        retain_loader = loaders['retain'],
        device        = device,
        epochs        = 10,
        lr            = 5e-4,
        alpha         = 0.5,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\n[Evaluating GP-Unlearn] …")
    gp_metrics = evaluate(gp_model, loaders, device)
    print(f"  {gp_metrics}")

    print("\n[Evaluating Original] …")
    original_metrics = evaluate(original, loaders, device)

    # ── Load prior results for comparison ─────────────────────────────────────
    week2 = {}
    if os.path.exists(W2_RESULTS_PATH):
        with open(W2_RESULTS_PATH) as f:
            week2 = json.load(f)
        print(f"[Week 2 results] Loaded ← {W2_RESULTS_PATH}")
    else:
        print(f"[WARNING] Week 2 results not found: {W2_RESULTS_PATH}")

    week3 = {}
    if os.path.exists(W3_RESULTS_PATH):
        with open(W3_RESULTS_PATH) as f:
            week3 = json.load(f)
        print(f"[Week 3 results] Loaded ← {W3_RESULTS_PATH}")
    else:
        print(f"[WARNING] Week 3 results not found: {W3_RESULTS_PATH}")

    retrain_ref = week2.get('retrain', {
        'Acc_Dr': 98.32, 'Acc_Df': 0.0, 'Acc_val': 79.76, 'MIA': 53.90
    })

    all_results = {}
    for key, label in [('original', 'Original'), ('retrain', 'Retrain (gold std)'),
                       ('finetune', 'Fine-tune'), ('neggrad', 'NegGrad')]:
        if key in week2:
            all_results[label] = week2[key]
    if 'dkf' in week3:
        all_results['DKF (Ours)'] = week3['dkf']
    all_results['GP-Unlearn (Ours)'] = gp_metrics

    # ── Print comparison table ─────────────────────────────────────────────────
    header    = f"\n{'Method':<22}  {'Acc_Dr':>7}  {'Acc_Df':>7}  {'Acc_val':>8}  {'MIA':>7}  {'Avg.Gap':>8}"
    separator = '-' * 70
    print(header)
    print(separator)
    lines = [header, separator]

    for name, m in all_results.items():
        if name in ('Original', 'Retrain (gold std)'):
            gap_str = '   —   '
        else:
            gap_str = f"{avg_gap(m, retrain_ref):>7.2f}%"
        row = (f"{name:<22}  {m['Acc_Dr']:>6.2f}%  {m['Acc_Df']:>6.2f}%  "
               f"{m['Acc_val']:>7.2f}%  {m['MIA']:>6.2f}%  {gap_str}")
        print(row)
        lines.append(row)
    print(separator)
    lines.append(separator)

    # ── Save ───────────────────────────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, 'projection_unlearning_results.json')
    with open(json_path, 'w') as f:
        json.dump({'gp_unlearn': gp_metrics, 'original': original_metrics}, f, indent=2)
    print(f"\n[Saved] {json_path}")

    table_path = os.path.join(RESULTS_DIR, 'projection_comparison_table.txt')
    with open(table_path, 'w') as f:
        f.write("Gradient Projection Unlearning — Comparison Table\n")
        f.write("=" * 70 + "\n")
        f.write('\n'.join(lines) + '\n')
    print(f"[Saved] {table_path}\n")


if __name__ == '__main__':
    main()
