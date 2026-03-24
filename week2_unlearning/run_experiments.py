"""
Week 2 — Main experiment runner.

Loads the Week 1 baseline model, applies all three unlearning methods,
evaluates each with four metrics, and prints a results table.

Usage:
    cd week2_unlearning
    uv run python run_experiments.py
    uv run python run_experiments.py --skip-retrain   # skip the slow retrain
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torchvision.models import resnet18

from config import (
    CHECKPOINT_DIR, CLASS_NAMES, FORGET_CLASS,
    NUM_CLASSES, ORIGINAL_MODEL_PATH, RESULTS_DIR,
    get_device, setup_dirs,
)
from data_utils import get_all_loaders
from evaluate import evaluate_model, print_results_table
from unlearn import finetune, negative_gradient, retrain


def load_original_model(device):
    """Load Week 1's trained model as the starting point."""
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Original model not found at '{ORIGINAL_MODEL_PATH}'.\n"
            f"Make sure Week 1 training is complete."
        )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    ckpt = torch.load(ORIGINAL_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    print(f"Loaded original model — epoch {ckpt['epoch']}, "
          f"test acc {ckpt['test_acc']:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-retrain', action='store_true',
                        help='Skip retraining from scratch (saves ~37 min)')
    parser.add_argument('--reuse-checkpoints', action='store_true',
                        help='Load all saved checkpoints, only redo NegGrad (saves ~45 min)')
    args = parser.parse_args()
    if args.reuse_checkpoints:
        args.skip_retrain = True

    setup_dirs()
    device = get_device()
    print(f"Device: {device}")
    print(f"Forget class: {CLASS_NAMES[FORGET_CLASS]} (class {FORGET_CLASS})\n")

    loaders        = get_all_loaders()
    original_model = load_original_model(device)

    results = {}

    # ── Original model (before any unlearning) ─────────────────────────────────
    print("\n[0/4] Evaluating original model (pre-unlearning) ...")
    results['Original'] = evaluate_model(original_model, loaders, device)

    # ── 1. Retrain ─────────────────────────────────────────────────────────────
    retrain_ckpt = os.path.join(CHECKPOINT_DIR, 'retrain_model.pth')
    if args.skip_retrain and os.path.exists(retrain_ckpt):
        print("\n[1/4] Loading cached Retrain model ...")
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        m.load_state_dict(torch.load(retrain_ckpt, map_location=device))
        retrain_model = m.to(device)
    else:
        print("\n[1/4] Retrain from scratch on D_r (gold standard) ...")
        retrain_model = retrain(loaders['retain'], device)
    results['Retrain'] = evaluate_model(retrain_model, loaders, device)

    # ── 2. Fine-tune ───────────────────────────────────────────────────────────
    ft_ckpt = os.path.join(CHECKPOINT_DIR, 'finetune_model.pth')
    if args.reuse_checkpoints and os.path.exists(ft_ckpt):
        print("\n[2/4] Loading cached Fine-tune model ...")
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        m.load_state_dict(torch.load(ft_ckpt, map_location=device))
        ft_model = m.to(device)
    else:
        print("\n[2/4] Fine-tune on D_r ...")
        ft_model = finetune(original_model, loaders['retain'], device)
    results['Fine-tune'] = evaluate_model(ft_model, loaders, device)

    # ── 3. Negative Gradient ───────────────────────────────────────────────────
    # Always retrain NegGrad — alpha changed from 0.5 → 0.1 to fix collapse
    print("\n[3/4] Negative Gradient (alpha=0.1, fixed from 0.5) ...")
    ng_model = negative_gradient(original_model, loaders['forget'],
                                 loaders['retain'], device)
    results['NegGrad'] = evaluate_model(ng_model, loaders, device)

    # ── Print results table ────────────────────────────────────────────────────
    print_results_table(results)

    # ── Save results to JSON ───────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, 'week2_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == '__main__':
    main()
