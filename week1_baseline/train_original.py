"""
Train a ResNet-18 model on CIFAR-10 (original model before unlearning).
Saves the best checkpoint to checkpoints/original_model.pth.

Usage:
    uv run python train_original.py
    uv run python train_original.py --epochs 30   # quick run (~88% acc)
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm

from config import (
    BATCH_SIZE, CHECKPOINT_DIR, CLASS_NAMES, EPOCHS_ORIGINAL,
    FORGET_CLASS, LR_ORIGINAL, MOMENTUM, NUM_CLASSES, WEIGHT_DECAY,
    get_device, setup_dirs,
)
from data_utils import get_all_loaders, print_split_info


# ── Model ──────────────────────────────────────────────────────────────────────

def build_resnet18(num_classes=NUM_CLASSES):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── Train / Evaluate helpers ───────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct    += outputs.argmax(1).eq(targets).sum().item()
        total      += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        correct    += outputs.argmax(1).eq(targets).sum().item()
        total      += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS_ORIGINAL,
                        help='Total epochs to train (default 100)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from existing checkpoint')
    args = parser.parse_args()

    setup_dirs()
    device  = get_device()
    print(f"Device: {device}")

    loaders = get_all_loaders(BATCH_SIZE)
    print_split_info(loaders, CLASS_NAMES, FORGET_CLASS)

    model     = build_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'original_model.pth')
    start_epoch = 1
    best_acc    = 0.0

    # CosineAnnealingLR always runs over the full T_max window so the LR
    # curve is correct whether we start fresh or resume mid-way.
    optimizer = optim.SGD(model.parameters(), lr=LR_ORIGINAL,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt['test_acc']
        # Fast-forward scheduler to match the already-completed epochs
        for _ in range(ckpt['epoch']):
            scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']}  (best so far: {best_acc:.2f}%)\n")
    else:
        print(f"Starting fresh training for {args.epochs} epochs ...\n")

    start = time.time()

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc='Training', unit='epoch'):
        train_loss, train_acc = train_one_epoch(
            model, loaders['train'], criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(
            model, loaders['test'],  criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'test_acc':   test_acc,
            }, ckpt_path)

        if epoch % 10 == 0:
            elapsed = time.time() - start
            tqdm.write(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"Train {train_acc:.1f}%  Test {test_acc:.1f}%  "
                f"Best {best_acc:.1f}%  [{elapsed:.0f}s]"
            )

    print(f"\nDone. Best test accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved → {ckpt_path}")


if __name__ == '__main__':
    main()
