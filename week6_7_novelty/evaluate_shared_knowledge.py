"""
Shared-knowledge evaluation for Week 6-7 novelty experiments.

The base paper reports accuracy, MIA, Avg.Gap, and qualitative plots. This file
adds representation-level evidence for the RA-DKF claim:

  - retain feature cosine similarity against teacher
  - retain feature drift score
  - teacher-student agreement
  - teacher-student KL divergence
  - clustering quality of student embeddings using ARI/NMI
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchvision.models import resnet18


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_W34 = os.path.join(_ROOT, "week3-4")
if _W34 not in sys.path:
    sys.path.insert(0, _W34)

import config as w34_config
import data_utils as w34_data_utils
from evaluate import evaluate_model
from ra_dkf import get_features


ORIGINAL_MODEL_PATH = os.path.join(_ROOT, "week1_baseline", "checkpoints", "original_model.pth")
RESULTS_DIR = os.path.join(_HERE, "results")
DATA_DIR = os.path.join(_ROOT, "week1_baseline", "data")


def load_resnet18(path, device):
    """Load a ResNet-18 checkpoint saved either as a raw state_dict or dict."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, w34_config.NUM_CLASSES)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def collect_features(model, loader, device, max_samples=5000):
    model.eval()
    features = []
    labels = []
    seen = 0

    for x, y in loader:
        remaining = max_samples - seen
        if remaining <= 0:
            break
        x = x[:remaining].to(device)
        y = y[:remaining]
        z = get_features(model, x).cpu()
        features.append(z)
        labels.append(y.cpu())
        seen += x.size(0)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def feature_drift_metrics(teacher, student, loader, device, max_samples=5000):
    # Teacher and student must process the SAME batches in the SAME order.
    # Calling collect_features twice on a shuffled loader gives different
    # sample orderings, making per-sample cosine similarity meaningless.
    teacher.eval()
    student.eval()
    zt_list, zs_list = [], []
    seen = 0

    for x, _ in loader:
        remaining = max_samples - seen
        if remaining <= 0:
            break
        x = x[:remaining].to(device)
        zt_list.append(get_features(teacher, x).cpu())
        zs_list.append(get_features(student, x).cpu())
        seen += x.size(0)

    z_teacher = torch.cat(zt_list, dim=0)
    z_student = torch.cat(zs_list, dim=0)

    z_teacher_n = F.normalize(z_teacher, dim=1)
    z_student_n = F.normalize(z_student, dim=1)
    cosine = F.cosine_similarity(z_teacher_n, z_student_n, dim=1)
    normalized_mse = F.mse_loss(z_student_n, z_teacher_n).item()

    return {
        "feature_cosine": float(cosine.mean().item()),
        "feature_drift": float(1.0 - cosine.mean().item()),
        "feature_mse_normalized": float(normalized_mse),
    }


@torch.no_grad()
def agreement_and_kl(teacher, student, loader, device, max_samples=5000):
    teacher.eval()
    student.eval()
    total = 0
    agree = 0
    kl_sum = 0.0

    for x, _ in loader:
        remaining = max_samples - total
        if remaining <= 0:
            break
        x = x[:remaining].to(device)
        t_logits = teacher(x)
        s_logits = student(x)

        agree += (t_logits.argmax(1) == s_logits.argmax(1)).sum().item()
        batch_kl = F.kl_div(
            F.log_softmax(s_logits, dim=1),
            F.softmax(t_logits, dim=1),
            reduction="batchmean",
        )
        kl_sum += batch_kl.item() * x.size(0)
        total += x.size(0)

    return {
        "agreement": 100.0 * agree / max(total, 1),
        "kl_teacher_student": kl_sum / max(total, 1),
    }


@torch.no_grad()
def clustering_metrics(model, loader, device, max_samples=5000):
    z, y = collect_features(model, loader, device, max_samples=max_samples)
    labels = y.numpy()
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return {"ari": 0.0, "nmi": 0.0}

    z_np = F.normalize(z, dim=1).numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = kmeans.fit_predict(z_np)
    return {
        "ari": float(adjusted_rand_score(labels, pred)),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
    }


def avg_gap(metrics, retrain_metrics):
    keys = ["Acc_Dr", "Acc_Df", "Acc_val", "MIA"]
    return sum(abs(metrics[k] - retrain_metrics[k]) for k in keys) / len(keys)


def evaluate_shared_knowledge(
    teacher,
    student,
    loaders,
    device,
    retrain_metrics=None,
    max_samples=5000,
):
    """Return standard unlearning metrics plus representation-level metrics."""
    standard = evaluate_model(student, loaders, device)
    out = dict(standard)

    retain_drift = feature_drift_metrics(
        teacher, student, loaders["retain"], device, max_samples=max_samples
    )
    forget_drift = feature_drift_metrics(
        teacher, student, loaders["forget"], device, max_samples=max_samples
    )
    retain_agree = agreement_and_kl(
        teacher, student, loaders["retain"], device, max_samples=max_samples
    )
    forget_agree = agreement_and_kl(
        teacher, student, loaders["forget"], device, max_samples=max_samples
    )
    retain_cluster = clustering_metrics(
        student, loaders["retain"], device, max_samples=max_samples
    )

    out.update({
        "Retain_Feature_Cosine": retain_drift["feature_cosine"],
        "Retain_Feature_Drift": retain_drift["feature_drift"],
        "Retain_Feature_MSE": retain_drift["feature_mse_normalized"],
        "Forget_Feature_Cosine": forget_drift["feature_cosine"],
        "Forget_Feature_Drift": forget_drift["feature_drift"],
        "Retain_Agreement": retain_agree["agreement"],
        "Retain_KL": retain_agree["kl_teacher_student"],
        "Forget_Agreement": forget_agree["agreement"],
        "Forget_KL": forget_agree["kl_teacher_student"],
        "Retain_ARI": retain_cluster["ari"],
        "Retain_NMI": retain_cluster["nmi"],
    })

    if retrain_metrics is not None:
        out["Avg.Gap"] = avg_gap(standard, retrain_metrics)

    return out


def print_shared_table(results):
    header = (
        f"{'Method':<18} {'Acc_Dr':>8} {'Acc_Df':>8} {'Acc_val':>8} "
        f"{'MIA':>8} {'AvgGap':>8} {'RetCos':>8} {'RetDrift':>9} {'RetAgr':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, m in results.items():
        gap = m.get("Avg.Gap", float("nan"))
        print(
            f"{name:<18} {m['Acc_Dr']:>7.2f}% {m['Acc_Df']:>7.2f}% "
            f"{m['Acc_val']:>7.2f}% {m['MIA']:>7.2f}% {gap:>7.2f}% "
            f"{m['Retain_Feature_Cosine']:>8.4f} "
            f"{m['Retain_Feature_Drift']:>9.4f} "
            f"{m['Retain_Agreement']:>7.2f}%"
        )


def _prepare_loaders():
    w34_config.DATA_DIR = DATA_DIR
    w34_data_utils.DATA_DIR = DATA_DIR
    return w34_data_utils.get_all_loaders()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--method-name", default="Student")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = w34_config.get_device()
    loaders = _prepare_loaders()

    teacher = load_resnet18(ORIGINAL_MODEL_PATH, device)
    student = load_resnet18(args.student_checkpoint, device)
    metrics = evaluate_shared_knowledge(
        teacher=teacher,
        student=student,
        loaders=loaders,
        device=device,
        max_samples=args.max_samples,
    )

    print_shared_table({args.method_name: metrics})

    out = args.output or os.path.join(RESULTS_DIR, f"{args.method_name}_shared_metrics.json")
    with open(out, "w") as f:
        json.dump({args.method_name: metrics}, f, indent=2)
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
