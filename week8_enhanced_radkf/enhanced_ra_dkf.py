"""
Week 8: Enhanced RA-DKF experiments.

This module is intentionally isolated from week6_7_novelty. It keeps the base
DKF objective but tests stronger representation-alignment variants:

1. Cosine alignment instead of normalized MSE.
2. Optional intermediate-layer alignment (layer2/layer3/avgpool).
3. Detached retain negatives in the contrastive loss, so the contrastive term
   moves forget/counterfactual representations without directly pulling retain
   features away from the teacher geometry.
"""
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_W34 = os.path.join(_ROOT, "week3-4")
if _W34 not in sys.path:
    sys.path.insert(0, _W34)

from beta_vae import BetaVAE
from config import (
    BETA,
    DKF_EPOCHS,
    DKF_LR,
    LAMBDA_C,
    LAMBDA_FORGET,
    LAMBDA_RETAIN,
    LATENT_DIM,
    NUM_CLASSES,
    TEMPERATURE,
    VAE_PRETRAIN_EPOCHS,
)


BASE_CHECKPOINT_DIR = os.path.join(_W34, "checkpoints")
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints")
VALID_ALIGN_LAYERS = ("layer1", "layer2", "layer3", "layer4", "avgpool")


def get_feature_maps(model, x, layers=("avgpool",)):
    """Extract pooled ResNet-18 features for one or more named stages."""
    requested = set(layers)
    unknown = requested.difference(VALID_ALIGN_LAYERS)
    if unknown:
        raise ValueError(f"Unknown align layer(s): {sorted(unknown)}")

    feats = {}

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    if "layer1" in requested:
        feats["layer1"] = F.adaptive_avg_pool2d(x, 1).flatten(1)

    x = model.layer2(x)
    if "layer2" in requested:
        feats["layer2"] = F.adaptive_avg_pool2d(x, 1).flatten(1)

    x = model.layer3(x)
    if "layer3" in requested:
        feats["layer3"] = F.adaptive_avg_pool2d(x, 1).flatten(1)

    x = model.layer4(x)
    if "layer4" in requested:
        feats["layer4"] = F.adaptive_avg_pool2d(x, 1).flatten(1)

    if "avgpool" in requested:
        feats["avgpool"] = model.avgpool(x).flatten(1)

    return feats


def get_features(model, x):
    """Compatibility helper used by evaluation: 512-dim avgpool features."""
    return get_feature_maps(model, x, layers=("avgpool",))["avgpool"]


def contrastive_loss(z_anchor, z_pos, z_neg, temperature=TEMPERATURE):
    """InfoNCE loss used by DKF for boundary rigidification."""
    z_anchor = F.normalize(z_anchor, dim=1)
    z_pos = F.normalize(z_pos, dim=1)
    z_neg = F.normalize(z_neg, dim=1)

    pos_sim = (z_anchor * z_pos).sum(dim=1, keepdim=True) / temperature
    neg_sim = torch.mm(z_anchor, z_neg.t()) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=z_anchor.device)
    return F.cross_entropy(logits, labels)


def cosine_alignment_loss(student_feats, teacher_feats, layers):
    """Mean 1-cosine distance across selected representation layers."""
    losses = []
    for layer in layers:
        z_s = F.normalize(student_feats[layer], dim=1)
        z_t = F.normalize(teacher_feats[layer], dim=1)
        losses.append(1.0 - F.cosine_similarity(z_s, z_t, dim=1).mean())
    return torch.stack(losses).mean()


def _cycle(loader):
    while True:
        yield from loader


def _vae_checkpoint_candidates(vae_pretrain_epochs, beta):
    name = f"vae_pretrained_e{vae_pretrain_epochs}_b{int(beta)}.pth"
    return [
        os.path.join(BASE_CHECKPOINT_DIR, name),
        os.path.join(CHECKPOINT_DIR, name),
    ]


def _load_or_train_vae(vae, forget_loader, retain_loader, device, epochs, beta, lr):
    """Reuse Week 3 VAE checkpoint if present; otherwise train a local copy."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    candidates = _vae_checkpoint_candidates(epochs, beta)
    for path in candidates:
        if os.path.exists(path):
            print(f"\n[Enhanced RA-DKF Phase 1] Loading beta-VAE checkpoint <- {path}")
            vae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            return vae

    save_path = candidates[-1]
    print(f"\n[Enhanced RA-DKF Phase 1] VAE checkpoint not found. Training for {epochs} epochs ...")
    vae_opt = optim.Adam(vae.parameters(), lr=lr)
    forget_cycle = _cycle(forget_loader)

    for _ in tqdm(range(1, epochs + 1), desc="Enhanced VAE pretrain", unit="epoch"):
        vae.train()
        for x_r, y_r in retain_loader:
            x_f, y_f = next(forget_cycle)
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)
            b = min(x_r.size(0), x_f.size(0))
            x_r, y_r, x_f, y_f = x_r[:b], y_r[:b], x_f[:b], y_f[:b]

            vae_opt.zero_grad()
            loss, _ = vae.compute_loss(x_f, x_r, y_r, y_f)
            loss.backward()
            vae_opt.step()

    torch.save(vae.state_dict(), save_path)
    print(f"[Enhanced RA-DKF Phase 1] beta-VAE saved -> {save_path}")
    return vae


def safe_name(value):
    if isinstance(value, (list, tuple)):
        value = "-".join(str(v) for v in value)
    return str(value).replace(".", "p").replace("-", "_").replace(",", "_")


def train_enhanced_ra_dkf(
    original_model,
    forget_loader,
    retain_loader,
    device,
    lambda_align=0.5,
    align_layers=("avgpool",),
    detach_retain_contrast=True,
    variant_name=None,
    vae_pretrain_epochs=VAE_PRETRAIN_EPOCHS,
    student_epochs=DKF_EPOCHS,
    lr=DKF_LR,
    lambda_retain=LAMBDA_RETAIN,
    lambda_forget=LAMBDA_FORGET,
    lambda_c=LAMBDA_C,
    max_train_batches=None,
):
    """
    Train an enhanced RA-DKF variant.

    The core experiment is whether a stronger cosine feature anchor, applied at
    final or intermediate ResNet stages, improves DKF's retain geometry without
    sacrificing forgetting.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    align_layers = tuple(align_layers)

    vae = BetaVAE(
        latent_dim_s=LATENT_DIM,
        latent_dim_u=LATENT_DIM,
        num_classes=NUM_CLASSES,
        beta=BETA,
    ).to(device)
    vae = _load_or_train_vae(
        vae=vae,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        device=device,
        epochs=vae_pretrain_epochs,
        beta=BETA,
        lr=lr,
    )
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    teacher = copy.deepcopy(original_model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = copy.deepcopy(original_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    forget_cycle = _cycle(forget_loader)

    print(
        f"\n[Enhanced RA-DKF Phase 2] Training for {student_epochs} epochs "
        f"(layers={align_layers}, lambda_align={lambda_align}, "
        f"detach_retain_contrast={detach_retain_contrast}, lr={lr}, "
        f"max_train_batches={max_train_batches})"
    )

    for epoch in tqdm(range(1, student_epochs + 1), desc="Enhanced RA-DKF", unit="epoch"):
        student.train()
        running = {"retain": 0.0, "forget": 0.0, "contrast": 0.0, "align": 0.0}
        steps = 0

        for x_r, y_r in retain_loader:
            x_f, _ = next(forget_cycle)
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f = x_f.to(device)
            b = min(x_r.size(0), x_f.size(0))
            x_r, y_r, x_f = x_r[:b], y_r[:b], x_f[:b]

            with torch.no_grad():
                _, x_cf, _, _, _, _, _, _ = vae(x_f, x_r)
                teacher_cf = F.softmax(teacher(x_cf), dim=1)
                teacher_align_feats = get_feature_maps(teacher, x_r, align_layers)

            optimizer.zero_grad()

            loss_retain = lambda_retain * criterion(student(x_r), y_r)
            loss_forget = lambda_forget * F.kl_div(
                F.log_softmax(student(x_f), dim=1),
                teacher_cf,
                reduction="batchmean",
            )

            z_f = get_features(student, x_f)
            z_cf = get_features(student, x_cf)
            z_r = get_features(student, x_r)
            z_r_contrast = z_r.detach() if detach_retain_contrast else z_r
            loss_contrast = lambda_c * contrastive_loss(z_f, z_cf, z_r_contrast)

            student_align_feats = get_feature_maps(student, x_r, align_layers)
            loss_align = lambda_align * cosine_alignment_loss(
                student_align_feats,
                teacher_align_feats,
                align_layers,
            )

            loss = loss_retain + loss_forget + loss_contrast + loss_align
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running["retain"] += loss_retain.item()
            running["forget"] += loss_forget.item()
            running["contrast"] += loss_contrast.item()
            running["align"] += loss_align.item()
            steps += 1
            if max_train_batches is not None and steps >= max_train_batches:
                break

        msg = "  ".join(f"{k}={v / max(steps, 1):.4f}" for k, v in running.items())
        tqdm.write(f"Epoch {epoch:2d}/{student_epochs}  {msg}")

    variant = variant_name or f"layers_{safe_name(align_layers)}_la_{safe_name(lambda_align)}"
    out = os.path.join(CHECKPOINT_DIR, f"enhanced_radkf_{variant}.pth")
    torch.save(student.state_dict(), out)
    print(f"\n[Enhanced RA-DKF] Saved -> {out}")
    return student, out
