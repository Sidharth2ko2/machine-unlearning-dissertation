"""
Data loading utilities for CIFAR-10 machine unlearning experiments.
Creates forget set (D_f) and retain set (D_r) based on FORGET_CLASS.
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # macOS cert fix

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from config import (BATCH_SIZE, DATA_DIR, FORGET_CLASS, NUM_CLASSES)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    return train_tf, test_tf


def get_datasets():
    train_tf, test_tf = get_transforms()
    train_ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_tf)
    return train_ds, test_ds


def split_forget_retain(train_ds):
    """Split training data into forget set (D_f) and retain set (D_r)."""
    targets = np.array(train_ds.targets)
    forget_idx  = np.where(targets == FORGET_CLASS)[0]
    retain_idx  = np.where(targets != FORGET_CLASS)[0]
    return Subset(train_ds, forget_idx), Subset(train_ds, retain_idx)


def get_all_loaders(batch_size=BATCH_SIZE):
    train_ds, test_ds = get_datasets()
    forget_ds, retain_ds = split_forget_retain(train_ds)

    loaders = {
        'train':  DataLoader(train_ds,  batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        'test':   DataLoader(test_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        'forget': DataLoader(forget_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        'retain': DataLoader(retain_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
    }
    return loaders


def print_split_info(loaders, class_names, forget_class):
    print(f"\nDataset: CIFAR-10")
    print(f"Forget class : {class_names[forget_class]} (class index {forget_class})")
    print(f"Train  size  : {len(loaders['train'].dataset):,}")
    print(f"Forget size  : {len(loaders['forget'].dataset):,}")
    print(f"Retain size  : {len(loaders['retain'].dataset):,}")
    print(f"Test   size  : {len(loaders['test'].dataset):,}\n")
