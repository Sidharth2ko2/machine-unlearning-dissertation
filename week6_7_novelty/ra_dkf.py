"""
Week 6-7 novelty: Representation-Aligned DKF (RA-DKF).

Base DKF preserves shared knowledge through β-VAE disentanglement and
counterfactual distillation. RA-DKF adds an explicit retain-feature alignment
term during student training:

    L_align = MSE(normalize(f_student(x_r)), normalize(f_teacher(x_r)))

where f(.) is the 512-dim ResNet-18 avgpool representation. This anchors the
student's retain-class representation geometry to the original teacher while
the forget loss redirects the target class.
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


def get_features(model, x):
    """Extract 512-dim avgpool features from a ResNet-18 model."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return x.flatten(1)


def contrastive_loss(z_anchor, z_pos, z_neg, temperature=TEMPERATURE):
    """InfoNCE loss used by base DKF for boundary rigidification."""
    z_anchor = F.normalize(z_anchor, dim=1)
    z_pos = F.normalize(z_pos, dim=1)
    z_neg = F.normalize(z_neg, dim=1)

    pos_sim = (z_anchor * z_pos).sum(dim=1, keepdim=True) / temperature
    neg_sim = torch.mm(z_anchor, z_neg.t()) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=z_anchor.device)
    return F.cross_entropy(logits, labels)


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
    """
    Reuse the Week 3 VAE checkpoint when available. If it is absent, train the
    same β-VAE and cache it locally under week6_7_novelty/checkpoints.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    candidates = _vae_checkpoint_candidates(epochs, beta)
    for path in candidates:
        if os.path.exists(path):
            print(f"\n[RA-DKF Phase 1] Loading β-VAE checkpoint ← {path}")
            vae.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            return vae

    save_path = candidates[-1]
    print(f"\n[RA-DKF Phase 1] β-VAE checkpoint not found. Training for {epochs} epochs ...")
    vae_opt = optim.Adam(vae.parameters(), lr=lr)
    forget_cycle = _cycle(forget_loader)

    for _ in tqdm(range(1, epochs + 1), desc="RA-DKF VAE pretrain", unit="epoch"):
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
    print(f"[RA-DKF Phase 1] β-VAE saved → {save_path}")
    return vae


def _safe_lambda_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def train_ra_dkf(
    original_model,
    forget_loader,
    retain_loader,
    device,
    lambda_align=0.5,
    vae_pretrain_epochs=VAE_PRETRAIN_EPOCHS,
    student_epochs=DKF_EPOCHS,
    lr=DKF_LR,
    lambda_retain=LAMBDA_RETAIN,
    lambda_forget=LAMBDA_FORGET,
    lambda_c=LAMBDA_C,
):
    """
    Train RA-DKF.

    RA-DKF keeps the base DKF objective and adds retain-class representation
    alignment against the frozen original model:

        L = L_retain + L_forget + L_contrast + λ_align * L_align
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
        f"\n[RA-DKF Phase 2] Training student for {student_epochs} epochs "
        f"(λ_retain={lambda_retain}, λ_forget={lambda_forget}, "
        f"λ_c={lambda_c}, λ_align={lambda_align}, lr={lr})"
    )

    for epoch in tqdm(range(1, student_epochs + 1), desc="RA-DKF student", unit="epoch"):
        student.train()
        running = {"retain": 0.0, "forget": 0.0, "contrast": 0.0, "align": 0.0}
        steps = 0

        for x_r, y_r in retain_loader:
            x_f, y_f = next(forget_cycle)
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f = x_f.to(device)
            b = min(x_r.size(0), x_f.size(0))
            x_r, y_r, x_f = x_r[:b], y_r[:b], x_f[:b]

            with torch.no_grad():
                _, x_cf, _, _, _, _, _, _ = vae(x_f, x_r)
                teacher_cf = F.softmax(teacher(x_cf), dim=1)
                z_teacher_r = F.normalize(get_features(teacher, x_r), dim=1)

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
            loss_contrast = lambda_c * contrastive_loss(z_f, z_cf, z_r)

            # Normalized feature MSE preserves geometry without over-penalizing scale.
            loss_align = lambda_align * F.mse_loss(F.normalize(z_r, dim=1), z_teacher_r)

            loss = loss_retain + loss_forget + loss_contrast + loss_align
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running["retain"] += loss_retain.item()
            running["forget"] += loss_forget.item()
            running["contrast"] += loss_contrast.item()
            running["align"] += loss_align.item()
            steps += 1

        msg = "  ".join(f"{k}={v / max(steps, 1):.4f}" for k, v in running.items())
        tqdm.write(f"Epoch {epoch:2d}/{student_epochs}  {msg}")

    out = os.path.join(CHECKPOINT_DIR, f"ra_dkf_align_{_safe_lambda_name(lambda_align)}.pth")
    torch.save(student.state_dict(), out)
    print(f"\n[RA-DKF] Saved → {out}")
    return student
