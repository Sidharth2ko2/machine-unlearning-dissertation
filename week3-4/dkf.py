"""
DKF — Disentangled Knowledge Forgetting (Stage 2: Knowledge Refine).

After the β-VAE generates counterfactual samples X_c = Decoder(S_f, U_r),
the student model (unlearned ResNet-18) is trained with three losses:

1. Retain loss   : CE(student(X_r), y_r)
   → Student must still classify the 9 retain classes correctly.

2. Forgetting loss L_f (Equation 10):
   → KL(teacher(X_c), student(X_f))
   → The student is told: "When you see an airplane (X_f), your output
     should match what the teacher predicts for the counterfactual (X_c)."
   → Since X_c looks like a retain-class sample (bird/ship), this forces
     the student to classify airplanes as retain-like → forgetting.

3. Contrastive loss L_c (Equation 12):
   → Pull: student embedding of X_c ↔ student embedding of X_r (same side)
   → Push: student embedding of X_c ↔ student embedding of X_f (opposite sides)
   → This rigidifies the decision boundary between forget and retain classes.

Total loss = L_retain + L_f + λ_c × L_c
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from beta_vae import BetaVAE
from config import (
    BETA, CHECKPOINT_DIR, DKF_EPOCHS, DKF_LR,
    LAMBDA_C, LAMBDA_FORGET, LAMBDA_RETAIN,
    LATENT_DIM, NUM_CLASSES, TEMPERATURE, VAE_PRETRAIN_EPOCHS,
)


# ── Feature extractor (penultimate layer of ResNet-18) ─────────────────────────

def get_features(model, x):
    """
    Extract 512-dim embeddings from ResNet-18's avgpool layer.
    Used for contrastive loss — we need the representation space,
    not the final class logits.
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return x.flatten(1)   # [B, 512]


# ── Contrastive loss (InfoNCE) ──────────────────────────────────────────────────

def contrastive_loss(z_anchor, z_pos, z_neg, temperature=TEMPERATURE):
    """
    InfoNCE loss for boundary rigidification.

    z_anchor : embeddings of counterfactual X_c   [B, 512]
    z_pos    : embeddings of retain samples X_r   [B, 512]
    z_neg    : embeddings of forget samples X_f   [B, 512]

    Goal: pull (anchor, positive) close, push (anchor, negative) apart.
    This makes the model's representation space clearly separate
    forget class from retain classes.
    """
    z_anchor = F.normalize(z_anchor, dim=1)
    z_pos    = F.normalize(z_pos,    dim=1)
    z_neg    = F.normalize(z_neg,    dim=1)

    # Positive similarity: anchor ↔ retain
    pos_sim = (z_anchor * z_pos).sum(dim=1, keepdim=True) / temperature   # [B, 1]

    # Negative similarity: anchor ↔ forget (all pairs)
    neg_sim = torch.mm(z_anchor, z_neg.t()) / temperature                  # [B, B]

    # InfoNCE: treat the paired positive as label 0
    logits = torch.cat([pos_sim, neg_sim], dim=1)                          # [B, 1+B]
    labels = torch.zeros(logits.size(0), dtype=torch.long,
                         device=z_anchor.device)

    return F.cross_entropy(logits, labels)


# ── DKF Training ───────────────────────────────────────────────────────────────

def train_dkf(original_model, forget_loader, retain_loader, device,
              vae_pretrain_epochs = VAE_PRETRAIN_EPOCHS,
              student_epochs      = DKF_EPOCHS,
              lr                  = DKF_LR,
              lambda_retain       = LAMBDA_RETAIN,
              lambda_forget       = LAMBDA_FORGET,
              lambda_c            = LAMBDA_C):
    """
    Full DKF training procedure — two phases:

    Phase 1 (VAE pre-training):
      Train only the β-VAE on forget+retain data so that counterfactual
      images X_c are stable and realistic before the student sees them.
      Jointly training from epoch 0 gave noisy X_c → bad distillation
      targets → Acc_Dr collapsed. Pre-training fixes this.

    Phase 2 (Student training, VAE frozen):
      Freeze the VAE. Train the student model with three losses:
        - Retain loss   (amplified by lambda_retain=2.0)
        - Forget loss   (distillation via counterfactuals)
        - Contrastive   (boundary rigidification, reduced weight 0.1)
    """

    def _cycle(loader):
        while True:
            yield from loader

    # ── Initialise β-VAE ──────────────────────────────────────────────────────
    vae = BetaVAE(
        latent_dim_s=LATENT_DIM,
        latent_dim_u=LATENT_DIM,
        num_classes=NUM_CLASSES,
        beta=BETA,
    ).to(device)

    # ── Teacher: frozen original model ────────────────────────────────────────
    teacher = copy.deepcopy(original_model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Pre-train β-VAE so counterfactuals are stable
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[DKF Phase 1] Pre-training β-VAE for {vae_pretrain_epochs} epochs ...")
    vae_opt      = optim.Adam(vae.parameters(), lr=lr)
    forget_cycle = _cycle(forget_loader)

    for epoch in tqdm(range(1, vae_pretrain_epochs + 1), desc='VAE pretrain', unit='epoch'):
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

    # Freeze VAE — counterfactuals are now fixed inputs for the student
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print("[DKF Phase 1] β-VAE pre-training complete. VAE frozen.")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Train student model with frozen VAE
    #
    # Paper's actual loss (Section 4.3, end): L = L_f + λ * L_c
    # There is NO explicit retain CE loss. Retain accuracy is preserved by
    # inertia — student starts from original model, only forget-class gradients
    # update it. Our previous retain CE was not in the paper and was the root
    # cause of Acc_Dr collapse (large λ_retain gradients corrupted backbone).
    #
    # Contrastive direction (paper Eq. 12):
    #   anchor = x_f  (forget sample)
    #   positive = x_c (counterfactual, same class as x_f)
    #   negative = x_r (retain samples, different class)
    # Previous code had this flipped (anchor=x_cf, pos=x_r, neg=x_f).
    #
    # Loop iterates over forget_loader (D_f, ~39 batches), cycling retain
    # samples as negatives — matching paper's "sum over x_f ∈ D_f" in Eq. 12.
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[DKF Phase 2] Training student for {student_epochs} epochs  "
          f"(λ_retain={lambda_retain}, λ_forget={lambda_forget}, "
          f"λ_c={lambda_c}, lr={lr}) ...")

    student      = copy.deepcopy(original_model).to(device)
    student_opt  = optim.Adam(student.parameters(), lr=lr)
    forget_cycle = _cycle(forget_loader)

    for epoch in tqdm(range(1, student_epochs + 1), desc='DKF student', unit='epoch'):
        student.train()

        for x_r, y_r in retain_loader:         # iterate over retain (strongest anchor)
            x_f, y_f = next(forget_cycle)
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)
            b = min(x_r.size(0), x_f.size(0))
            x_r, y_r, x_f, y_f = x_r[:b], y_r[:b], x_f[:b], y_f[:b]

            # Generate counterfactual with frozen VAE
            with torch.no_grad():
                _, x_cf, _, _, _, _, _, _ = vae(x_f, x_r)

            student_opt.zero_grad()

            # Loss 1: Retain CE — strongly anchor the 9 retain classes
            loss_retain = lambda_retain * criterion(student(x_r), y_r)

            # Loss 2: Forgetting via counterfactual distillation (L_f, Eq. 10)
            with torch.no_grad():
                teacher_cf = F.softmax(teacher(x_cf), dim=1)
            loss_forget = lambda_forget * F.kl_div(
                F.log_softmax(student(x_f), dim=1),
                teacher_cf, reduction='batchmean'
            )

            # Loss 3: Contrastive (L_c, Eq. 12) — paper's exact direction:
            #   anchor=x_f, positive=x_cf, negative=x_r
            # z_r is detached so gradient only flows through x_f and x_cf,
            # preventing contrastive from reshaping retain representations.
            z_f  = get_features(student, x_f)
            z_cf = get_features(student, x_cf)
            z_r  = get_features(student, x_r).detach()
            loss_contrast = lambda_c * contrastive_loss(z_f, z_cf, z_r)

            loss = loss_retain + loss_forget + loss_contrast
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            student_opt.step()

    path = os.path.join(CHECKPOINT_DIR, 'dkf_model.pth')
    torch.save(student.state_dict(), path)
    print(f"\n[DKF] Saved → {path}")
    return student
