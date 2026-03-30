"""
Week 3 — Run DKF and compare against Week 2 baselines.

Loads Week 2 results from JSON, runs DKF, and prints a full comparison
table with all methods side by side (matching paper's Table 1 format).

Usage:
    cd week3-4
    uv run python run_experiments.py
"""
import json
import os

import torch
import torch.nn as nn
from torchvision.models import resnet18

from config import (
    CHECKPOINT_DIR, CLASS_NAMES, FORGET_CLASS,
    NUM_CLASSES, ORIGINAL_MODEL_PATH, RESULTS_DIR,
    WEEK2_RESULTS_PATH, get_device, setup_dirs,
)
from data_utils import get_all_loaders
from dkf import train_dkf
from evaluate import evaluate_model, print_results_table


def load_model(path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    ckpt = torch.load(path, map_location=device)
    # handle both raw state_dict and wrapped checkpoint
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    return model.to(device)


def main():
    setup_dirs()
    device = get_device()
    print(f"Device  : {device}")
    print(f"Forget  : {CLASS_NAMES[FORGET_CLASS]} (class {FORGET_CLASS})\n")

    loaders = get_all_loaders()

    # ── Load original model ───────────────────────────────────────────────────
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Original model not found at '{ORIGINAL_MODEL_PATH}'.\n"
            "Run Week 1 training first."
        )
    ckpt = torch.load(ORIGINAL_MODEL_PATH, map_location=device)
    original_model = resnet18(weights=None)
    original_model.fc = nn.Linear(original_model.fc.in_features, NUM_CLASSES)
    original_model.load_state_dict(ckpt['model_state_dict'])
    original_model = original_model.to(device)
    print(f"Loaded original model — epoch {ckpt['epoch']}, "
          f"test acc {ckpt['test_acc']:.2f}%\n")

    # ── Load Week 2 baseline results ──────────────────────────────────────────
    results = {}
    if os.path.exists(WEEK2_RESULTS_PATH):
        with open(WEEK2_RESULTS_PATH) as f:
            results = json.load(f)
        print(f"Loaded Week 2 results from {WEEK2_RESULTS_PATH}")
    else:
        print("Week 2 results not found — run Week 2 experiments first.")

    # ── Run DKF ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running DKF — Disentangled Knowledge Forgetting")
    print("=" * 60)
    dkf_model = train_dkf(
        original_model=original_model,
        forget_loader=loaders['forget'],
        retain_loader=loaders['retain'],
        device=device,
    )

    print("\nEvaluating DKF model ...")
    results['DKF (Ours)'] = evaluate_model(dkf_model, loaders, device)

    # ── Print full comparison table ───────────────────────────────────────────
    print_results_table(results)

    # ── Save results ──────────────────────────────────────────────────────────
    out = os.path.join(RESULTS_DIR, 'week3_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == '__main__':
    main()
