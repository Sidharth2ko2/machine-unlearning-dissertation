"""
Week 6-7 experiment runner for RA-DKF.

Runs a λ_align sweep and compares RA-DKF against base DKF using:
  - standard unlearning metrics: Acc_Dr, Acc_Df, Acc_val, MIA, Avg.Gap
  - representation metrics: retain feature cosine/drift, agreement, ARI/NMI

Usage:
    cd week6_7_novelty
    uv run python run_experiments.py --reuse-checkpoints
    uv run python run_experiments.py --lambda-aligns 0.1 0.5 1.0
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torchvision.models import resnet18


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_W34 = os.path.join(_ROOT, "week3-4")
_W2 = os.path.join(_ROOT, "week2_unlearning")
if _W34 not in sys.path:
    sys.path.insert(0, _W34)

import config as w34_config
import data_utils as w34_data_utils
from evaluate_shared_knowledge import evaluate_shared_knowledge, print_shared_table
from ra_dkf import CHECKPOINT_DIR, train_ra_dkf


DATA_DIR = os.path.join(_ROOT, "week1_baseline", "data")
ORIGINAL_MODEL_PATH = os.path.join(_ROOT, "week1_baseline", "checkpoints", "original_model.pth")
BASE_DKF_PATH = os.path.join(_W34, "checkpoints", "dkf_model.pth")
WEEK2_RESULTS_PATH = os.path.join(_W2, "results", "week2_results.json")
RESULTS_DIR = os.path.join(_HERE, "results")


def safe_lambda_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def load_resnet18(path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, w34_config.NUM_CLASSES)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model.to(device)


def load_original_model(device):
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        raise FileNotFoundError(
            f"Original model not found: {ORIGINAL_MODEL_PATH}\n"
            "Run week1_baseline/train_original.py first."
        )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, w34_config.NUM_CLASSES)
    ckpt = torch.load(ORIGINAL_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"[Loaded] Original model epoch={ckpt['epoch']} acc={ckpt['test_acc']:.2f}%")
    return model


def prepare_loaders():
    # data_utils imported DATA_DIR by value from config, so patch both modules.
    w34_config.DATA_DIR = DATA_DIR
    w34_data_utils.DATA_DIR = DATA_DIR
    return w34_data_utils.get_all_loaders()


def load_retrain_reference():
    if not os.path.exists(WEEK2_RESULTS_PATH):
        print(f"[Warning] Week 2 results missing: {WEEK2_RESULTS_PATH}")
        print("[Warning] Avg.Gap will use the documented Week 2 retrain fallback.")
        return {"Acc_Dr": 98.32, "Acc_Df": 0.0, "Acc_val": 79.76, "MIA": 53.90}

    with open(WEEK2_RESULTS_PATH) as f:
        week2 = json.load(f)

    return week2.get("Retrain") or week2.get("retrain")


def maybe_load_base_dkf(device):
    if not os.path.exists(BASE_DKF_PATH):
        print(f"[Warning] Base DKF checkpoint missing: {BASE_DKF_PATH}")
        print("[Warning] Run week3-4/run_experiments.py to include DKF comparison.")
        return None
    return load_resnet18(BASE_DKF_PATH, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lambda-aligns",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="λ_align values to test for RA-DKF.",
    )
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Load existing RA-DKF checkpoints if present.",
    )
    parser.add_argument("--student-epochs", type=int, default=w34_config.DKF_EPOCHS)
    parser.add_argument("--max-eval-samples", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = w34_config.get_device()
    print(f"[Device] {device}")
    print(f"[Forget class] {w34_config.CLASS_NAMES[w34_config.FORGET_CLASS]} ({w34_config.FORGET_CLASS})")

    loaders = prepare_loaders()
    original_model = load_original_model(device)
    retrain_ref = load_retrain_reference()

    results = {}

    base_dkf = maybe_load_base_dkf(device)
    if base_dkf is not None:
        print("\n[Evaluating] Base DKF")
        results["DKF"] = evaluate_shared_knowledge(
            teacher=original_model,
            student=base_dkf,
            loaders=loaders,
            device=device,
            retrain_metrics=retrain_ref,
            max_samples=args.max_eval_samples,
        )

    for lam in args.lambda_aligns:
        method = f"RA-DKF λ={lam:g}"
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ra_dkf_align_{safe_lambda_name(lam)}.pth")

        if args.reuse_checkpoints and os.path.exists(ckpt_path):
            print(f"\n[Loading] {method} ← {ckpt_path}")
            model = load_resnet18(ckpt_path, device)
        else:
            print(f"\n[Training] {method}")
            model = train_ra_dkf(
                original_model=original_model,
                forget_loader=loaders["forget"],
                retain_loader=loaders["retain"],
                device=device,
                lambda_align=lam,
                student_epochs=args.student_epochs,
            )

        print(f"\n[Evaluating] {method}")
        results[method] = evaluate_shared_knowledge(
            teacher=original_model,
            student=model,
            loaders=loaders,
            device=device,
            retrain_metrics=retrain_ref,
            max_samples=args.max_eval_samples,
        )

    print_shared_table(results)

    out = os.path.join(RESULTS_DIR, "week6_7_ra_dkf_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out}")

    best_name, best_metrics = min(results.items(), key=lambda item: item[1].get("Avg.Gap", float("inf")))
    print(
        f"[Best Avg.Gap] {best_name}: {best_metrics['Avg.Gap']:.2f}% "
        f"(Retain drift={best_metrics['Retain_Feature_Drift']:.4f})"
    )


if __name__ == "__main__":
    main()
