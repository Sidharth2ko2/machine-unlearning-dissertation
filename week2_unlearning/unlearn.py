"""
Week 2 — Baseline Unlearning Methods

Three methods implemented:
  1. Retrain      — train fresh model only on D_r (gold standard)
  2. Fine-tune    — continue training original model on D_r only
  3. NegGrad      — gradient ascent on D_f + gradient descent on D_r
"""
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR, EPOCHS_RETRAIN, EPOCHS_UNLEARN,
    LR_ORIGINAL, LR_FINETUNE, LR_NEGGRAD, MOMENTUM, NEGGRAD_ALPHA,
    NUM_CLASSES, WEIGHT_DECAY,
)


def _build_resnet18(num_classes=NUM_CLASSES):
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _cycle(loader):
    """Cycle through a dataloader indefinitely."""
    while True:
        yield from loader


# ── 1. Retrain ─────────────────────────────────────────────────────────────────

def retrain(retain_loader, device, epochs=EPOCHS_RETRAIN):
    """
    Gold standard: train a completely new ResNet-18 only on the retain set D_r.
    The model never sees any airplane image — perfect unlearning by definition.
    Used as the reference that all other methods are benchmarked against.
    """
    print(f"\n[Retrain] Training fresh ResNet-18 on D_r only ({epochs} epochs) ...")
    model     = _build_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR_ORIGINAL,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in tqdm(range(1, epochs + 1), desc='Retrain', unit='epoch'):
        model.train()
        for inputs, targets in retain_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            criterion(model(inputs), targets).backward()
            optimizer.step()
        scheduler.step()

    path = os.path.join(CHECKPOINT_DIR, 'retrain_model.pth')
    torch.save(model.state_dict(), path)
    print(f"[Retrain] Saved → {path}")
    return model


# ── 2. Fine-tune ───────────────────────────────────────────────────────────────

def finetune(original_model, retain_loader, device, epochs=EPOCHS_UNLEARN):
    """
    Fine-tune the original model on D_r only.
    The model gradually overwrites airplane knowledge since it never
    sees airplane images during fine-tuning. Simple and fast, but
    may not fully erase forget-class knowledge from the weights.
    """
    print(f"\n[Fine-tune] Fine-tuning on D_r only ({epochs} epochs) ...")
    model     = copy.deepcopy(original_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR_FINETUNE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for epoch in tqdm(range(1, epochs + 1), desc='Fine-tune', unit='epoch'):
        model.train()
        for inputs, targets in retain_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            criterion(model(inputs), targets).backward()
            optimizer.step()

    path = os.path.join(CHECKPOINT_DIR, 'finetune_model.pth')
    torch.save(model.state_dict(), path)
    print(f"[Fine-tune] Saved → {path}")
    return model


# ── 3. Negative Gradient ───────────────────────────────────────────────────────

def negative_gradient(original_model, forget_loader, retain_loader,
                      device, epochs=EPOCHS_UNLEARN, alpha=NEGGRAD_ALPHA):
    """
    Explicitly forget D_f via gradient ascent while retaining D_r via
    gradient descent. Combined loss = L_retain - alpha * L_forget.

    alpha controls the trade-off:
      - Higher alpha → more aggressive forgetting (risks hurting retain acc)
      - Lower alpha  → gentler forgetting (may be incomplete)

    Gradient clipping (max_norm=1.0) prevents the ascent from
    destabilizing the model weights.
    """
    print(f"\n[NegGrad] Gradient ascent on D_f + descent on D_r "
          f"({epochs} epochs, alpha={alpha}, lr={LR_NEGGRAD}) ...")
    model     = copy.deepcopy(original_model)
    criterion = nn.CrossEntropyLoss()
    # Use a much smaller LR than fine-tune — gradient ascent is inherently
    # unstable and a high LR destroys shared retain/forget features.
    optimizer = optim.SGD(model.parameters(), lr=LR_NEGGRAD,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    forget_cycle = _cycle(forget_loader)

    for epoch in tqdm(range(1, epochs + 1), desc='NegGrad', unit='epoch'):
        model.train()
        for inputs_r, targets_r in retain_loader:
            inputs_r, targets_r = inputs_r.to(device), targets_r.to(device)
            inputs_f, targets_f = next(forget_cycle)
            inputs_f, targets_f = inputs_f.to(device), targets_f.to(device)

            optimizer.zero_grad()
            loss_retain = criterion(model(inputs_r), targets_r)
            loss_forget = criterion(model(inputs_f), targets_f)
            # Descend on retain, ascend on forget
            loss = loss_retain - alpha * loss_forget
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    path = os.path.join(CHECKPOINT_DIR, 'neggrad_model.pth')
    torch.save(model.state_dict(), path)
    print(f"[NegGrad] Saved → {path}")
    return model
