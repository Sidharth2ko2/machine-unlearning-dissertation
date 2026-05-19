"""
Week 8 experiment runner for enhanced RA-DKF variants.

This folder is isolated from week6_7_novelty. It reuses the existing Week 1
original model and Week 3 DKF/VAE checkpoints, then trains only new Week 8
checkpoints under week8_enhanced_radkf/checkpoints/.

Examples:
    cd week8_enhanced_radkf
    uv run python run_experiments.py --student-epochs 1 --max-eval-samples 1000
    uv run python run_experiments.py --variants avgpool layer3 layer2-layer3 --lambda-aligns 0.1 0.5 1.0
    uv run python run_experiments.py --reuse-checkpoints
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_W34 = os.path.join(_ROOT, "week3-4")
_W2 = os.path.join(_ROOT, "week2_unlearning")
if _W34 not in sys.path:
    sys.path.insert(0, _W34)

import config as w34_config
import data_utils as w34_data_utils
from enhanced_ra_dkf import CHECKPOINT_DIR, safe_name, train_enhanced_ra_dkf
from evaluate_shared_knowledge import evaluate_shared_knowledge, print_shared_table


DATA_DIR = os.path.join(_ROOT, "week1_baseline", "data")
ORIGINAL_MODEL_PATH = os.path.join(_ROOT, "week1_baseline", "checkpoints", "original_model.pth")
BASE_DKF_PATH = os.path.join(_W34, "checkpoints", "dkf_model.pth")
WEEK2_RESULTS_PATH = os.path.join(_W2, "results", "week2_results.json")
RESULTS_DIR = os.path.join(_HERE, "results")

VARIANT_LAYERS = {
    "avgpool": ("avgpool",),
    "layer3": ("layer3",),
    "layer2": ("layer2",),
    "layer2-layer3": ("layer2", "layer3"),
    "layer3-avgpool": ("layer3", "avgpool"),
}


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
            "Copy week1_baseline/checkpoints/original_model.pth first."
        )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, w34_config.NUM_CLASSES)
    ckpt = torch.load(ORIGINAL_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"[Loaded] Original model epoch={ckpt['epoch']} acc={ckpt['test_acc']:.2f}%")
    return model


def prepare_loaders():
    w34_config.DATA_DIR = DATA_DIR
    w34_data_utils.DATA_DIR = DATA_DIR
    loaders = w34_data_utils.get_all_loaders()
    # macOS sandboxed runs can block torch_shm_manager used by worker processes.
    # Rebuild loaders locally with workers disabled for Week 8 experiments.
    return {
        key: DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=key in {"train", "forget", "retain"},
            num_workers=0,
        )
        for key, loader in loaders.items()
    }


def load_retrain_reference():
    if not os.path.exists(WEEK2_RESULTS_PATH):
        print(f"[Warning] Week 2 results missing: {WEEK2_RESULTS_PATH}")
        return {"Acc_Dr": 98.32, "Acc_Df": 0.0, "Acc_val": 79.76, "MIA": 53.90}

    with open(WEEK2_RESULTS_PATH) as f:
        week2 = json.load(f)
    return week2.get("Retrain") or week2.get("retrain")


def maybe_load_base_dkf(device):
    if not os.path.exists(BASE_DKF_PATH):
        print(f"[Warning] Base DKF checkpoint missing: {BASE_DKF_PATH}")
        return None
    return load_resnet18(BASE_DKF_PATH, device)


def checkpoint_name(variant, lambda_align, detach_retain_contrast):
    detach = "detach" if detach_retain_contrast else "nodetach"
    return f"enhanced_radkf_{variant}_la_{safe_name(lambda_align)}_{detach}.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["avgpool", "layer3", "layer2-layer3"],
        choices=sorted(VARIANT_LAYERS),
        help="Enhanced alignment variants to test.",
    )
    parser.add_argument(
        "--lambda-aligns",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="lambda_align values to sweep.",
    )
    parser.add_argument(
        "--lambda-forgets",
        type=float,
        nargs="+",
        default=[w34_config.LAMBDA_FORGET],
        help="lambda_forget values to sweep. Base DKF uses 0.06.",
    )
    parser.add_argument(
        "--lambda-c",
        type=float,
        default=w34_config.LAMBDA_C,
        help="contrastive loss weight.",
    )
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Load existing Week 8 checkpoints if present.",
    )
    parser.add_argument(
        "--no-detach-retain-contrast",
        action="store_true",
        help="Ablation: allow contrastive gradients through retain features.",
    )
    parser.add_argument("--student-epochs", type=int, default=w34_config.DKF_EPOCHS)
    parser.add_argument("--max-eval-samples", type=int, default=5000)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Debug only: cap retain-loop batches per epoch.",
    )
    parser.add_argument(
        "--quick-eval",
        action="store_true",
        help="Debug only: limit standard Acc/MIA metrics to max-eval-samples.",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    detach_retain_contrast = not args.no_detach_retain_contrast

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
            standard_max_samples=args.max_eval_samples if args.quick_eval else None,
        )

    for variant in args.variants:
        layers = VARIANT_LAYERS[variant]
        for lam in args.lambda_aligns:
            for lambda_forget in args.lambda_forgets:
                method = f"E-RA-DKF {variant} la={lam:g} lf={lambda_forget:g}"
                forget_name = f"lf_{safe_name(lambda_forget)}"
                ckpt_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"enhanced_radkf_{variant}_la_{safe_name(lam)}_{forget_name}_{'detach' if detach_retain_contrast else 'nodetach'}.pth",
                )

                if args.reuse_checkpoints and os.path.exists(ckpt_path):
                    print(f"\n[Loading] {method} <- {ckpt_path}")
                    model = load_resnet18(ckpt_path, device)
                else:
                    print(f"\n[Training] {method}")
                    model, saved_path = train_enhanced_ra_dkf(
                        original_model=original_model,
                        forget_loader=loaders["forget"],
                        retain_loader=loaders["retain"],
                        device=device,
                        lambda_align=lam,
                        align_layers=layers,
                        detach_retain_contrast=detach_retain_contrast,
                        variant_name=f"{variant}_la_{safe_name(lam)}_{forget_name}_{'detach' if detach_retain_contrast else 'nodetach'}",
                        student_epochs=args.student_epochs,
                        lambda_forget=lambda_forget,
                        lambda_c=args.lambda_c,
                        max_train_batches=args.max_train_batches,
                    )
                    ckpt_path = saved_path

                print(f"\n[Evaluating] {method}")
                results[method] = evaluate_shared_knowledge(
                    teacher=original_model,
                    student=model,
                    loaders=loaders,
                    device=device,
                    retrain_metrics=retrain_ref,
                    max_samples=args.max_eval_samples,
                    standard_max_samples=args.max_eval_samples if args.quick_eval else None,
                )
                results[method]["Checkpoint"] = ckpt_path
                results[method]["Align_Layers"] = list(layers)
                results[method]["Detach_Retain_Contrast"] = detach_retain_contrast
                results[method]["Lambda_Align"] = lam
                results[method]["Lambda_Forget"] = lambda_forget
                results[method]["Lambda_C"] = args.lambda_c

    print_shared_table(results)

    out = os.path.join(RESULTS_DIR, "week8_enhanced_radkf_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out}")

    best_name, best_metrics = min(results.items(), key=lambda item: item[1].get("Avg.Gap", float("inf")))
    print(
        f"[Best Avg.Gap] {best_name}: {best_metrics['Avg.Gap']:.2f}% "
        f"(Retain drift={best_metrics['Retain_Feature_Drift']:.4f}, "
        f"MIA={best_metrics['MIA']:.2f}%)"
    )


if __name__ == "__main__":
    main()
